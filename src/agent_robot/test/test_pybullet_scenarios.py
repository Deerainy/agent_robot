"""DIRECT-mode tests for fixed and parameterized PyBullet scenes."""

import os
import sys

import pybullet as p
import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot import pybullet_robot  # noqa: E402
from agent_robot.scenarios.generator import generate_scenario  # noqa: E402
from agent_robot.scene_graph import (  # noqa: E402
    SceneGraph,
    validate_actions,
)
from agent_robot.scenarios.spec import (  # noqa: E402
    CATEGORY_CLASSIFICATION,
    CATEGORY_RECOVERY,
    CATEGORY_SAFE_REJECTION,
    CATEGORY_SEQUENTIAL,
    CATEGORY_TARGET_PICK,
    REJECT_NOT_GRASPABLE,
    REJECT_WRONG_CONTAINER,
)
from agent_robot.scenarios.world_model import WorldModel  # noqa: E402
from agent_robot.skill_registry import (  # noqa: E402
    SkillRegistry,
    parse_actions,
)


def relocation_actions(object_id, container_id):
    return [
        {'skill': 'move_to', 'object': object_id},
        {'skill': 'pick', 'object': object_id},
        {'skill': 'move_to', 'object': container_id},
        {
            'skill': 'place',
            'object': object_id,
            'target': container_id,
        },
    ]


def run_parameterized_actions(spec, raw_actions):
    robot_id, objects = create_direct_scene(spec)
    state = {'held_object': None, 'constraint_id': None}
    registry = SkillRegistry()
    for name, handler in pybullet_robot.build_skill_handlers(
        robot_id, objects, state
    ).items():
        registry.register(name, handler)
    for action in parse_actions(raw_actions):
        registry.execute(action)
    return objects


def create_direct_scene(spec=None):
    return pybullet_robot.create_scene(
        spec=spec,
        connection_mode=p.DIRECT,
    )


def test_legacy_scene_still_contains_original_objects():
    robot_id, objects = create_direct_scene()
    try:
        assert p.isConnected()
        assert robot_id >= 0
        assert set(objects) == {'apple', 'cup', 'basket'}
        assert objects['apple'].graspable is True
        assert objects['basket'].receptacle == 'container'
        assert p.getNumBodies() == 10
        state = {'held_object': None, 'constraint_id': None}
        registry = SkillRegistry()
        for name, handler in pybullet_robot.build_skill_handlers(
            robot_id, objects, state
        ).items():
            registry.register(name, handler)
        for action in parse_actions(
            relocation_actions('apple', 'basket')
        ):
            registry.execute(action)
        position = p.getBasePositionAndOrientation(
            objects['apple'].body_id
        )[0]
        assert abs(position[0] - objects['basket'].position[0]) < 0.10
        assert abs(position[1] - objects['basket'].position[1]) < 0.07
    finally:
        p.disconnect()


def test_parameterized_scene_creates_colored_instances_and_open_bins():
    spec = generate_scenario(CATEGORY_CLASSIFICATION, seed=11)
    robot_id, objects = create_direct_scene(spec)
    try:
        assert robot_id >= 0
        assert set(objects) == {
            instance.instance_id for instance in spec.objects
        }
        for instance in spec.objects:
            object_spec = objects[instance.instance_id]
            assert object_spec.object_type == instance.object_type
            assert object_spec.position == list(instance.position)
            if instance.object_type == 'block':
                assert len(object_spec.body_ids) == 1
                visual = p.getVisualShapeData(object_spec.body_id)[0]
                assert list(visual[7]) == pytest.approx(
                    pybullet_robot._COLOR_RGBA[instance.color],
                    abs=1e-6,
                )
            elif instance.object_type == 'bin':
                assert object_spec.body_id == -1
                assert len(object_spec.body_ids) == 5
                assert object_spec.place_rest_position is not None
    finally:
        p.disconnect()


def test_world_slide_event_moves_body_and_returns_actual_pose():
    spec = generate_scenario(CATEGORY_RECOVERY, seed=7)
    _, objects = create_direct_scene(spec)
    try:
        target_id = spec.goal.params['target_id']
        target = objects[target_id]
        event = next(
            event for event in spec.events if event.target == target_id
        )

        acknowledgement = pybullet_robot.apply_world_event(
            objects,
            {
                'revision': 1,
                'event': event.event,
                'target': event.target,
                'new_position': list(event.new_position),
                'new_support': event.new_support,
            },
        )

        assert acknowledgement['revision'] == 1
        assert acknowledgement['applied'] is True
        actual = acknowledgement['actual_position']
        assert abs(actual[0] - event.new_position[0]) < 0.03
        assert abs(actual[1] - event.new_position[1]) < 0.03
        assert target.position == actual
        assert target.grasp_position[0] == actual[0]
        assert target.above_position[1] == actual[1]
    finally:
        p.disconnect()


def test_physical_world_model_commits_after_direct_event_ack():
    spec = generate_scenario(CATEGORY_RECOVERY, seed=7)
    world = WorldModel(spec, physical=True)
    target_id = spec.goal.params['target_id']
    event = spec.events[0]
    world.on_action_completed('move_to', target_id, None, 1)
    command = world.drain_commands()[0]
    _, objects = create_direct_scene(spec)
    try:
        acknowledgement = pybullet_robot.apply_world_event(
            objects,
            {
                'revision': command.revision,
                'event': command.event,
                'target': command.target,
                'new_position': list(command.new_position),
                'new_support': command.new_support,
            },
        )
        actual_position = tuple(acknowledgement['actual_position'])
        assert world.revision == 0
        assert world.on_event_ack(
            command.revision,
            acknowledgement['applied'],
            actual_position,
        )
        assert world.revision == 1
        assert world.instances[target_id].position == actual_position
        assert world.instances[target_id].position[0] == pytest.approx(
            event.new_position[0], abs=0.03
        )
    finally:
        p.disconnect()


def test_direct_recovery_rejects_stale_pick_then_replans_successfully():
    spec = generate_scenario(CATEGORY_RECOVERY, seed=7)
    world = WorldModel(spec, physical=True)
    graph = SceneGraph.build_from_environment(world.environment_payload())
    target = spec.goal.params['target_id']
    container = spec.goal.params['container_id']
    stale_actions = parse_actions(
        relocation_actions(target, container)
    )
    robot_id, objects = create_direct_scene(spec)
    state = {'held_object': None, 'constraint_id': None}
    registry = SkillRegistry()
    for name, handler in pybullet_robot.build_skill_handlers(
        robot_id, objects, state
    ).items():
        registry.register(name, handler)

    try:
        assert validate_actions(stale_actions, graph).ok
        registry.execute(stale_actions[0])
        world.on_action_completed('move_to', target, None, 1)
        command = world.drain_commands()[0]
        acknowledgement = pybullet_robot.apply_world_event(
            objects,
            {
                'revision': command.revision,
                'event': command.event,
                'target': command.target,
                'new_position': list(command.new_position),
                'new_support': command.new_support,
            },
        )
        assert acknowledgement['applied'] is True
        assert world.on_event_ack(
            command.revision,
            True,
            tuple(acknowledgement['actual_position']),
        )
        assert graph.update_observation(world.environment_payload())
        stale_report = validate_actions(stale_actions[1:], graph)
        assert stale_report.ok is False
        assert stale_report.reason_code == 'robot_not_at_object'

        replanned = parse_actions(
            relocation_actions(target, container)
        )
        assert validate_actions(replanned, graph).ok
        for action in replanned:
            registry.execute(action)

        final_position = p.getBasePositionAndOrientation(
            objects[target].body_id
        )[0]
        container_position = objects[container].position
        assert abs(final_position[0] - container_position[0]) < 0.10
        assert abs(final_position[1] - container_position[1]) < 0.07
    finally:
        p.disconnect()


def test_invalid_grasp_and_container_plans_are_rejected_without_motion():
    cases = (
        (
            REJECT_WRONG_CONTAINER,
            [
                {'skill': 'move_to', 'object': 'red_block_1'},
                {'skill': 'pick', 'object': 'red_block_1'},
                {'skill': 'move_to', 'object': 'cup'},
                {
                    'skill': 'place',
                    'object': 'red_block_1',
                    'target': 'cup',
                },
            ],
            'target_not_receptacle',
        ),
        (
            REJECT_NOT_GRASPABLE,
            [
                {'skill': 'move_to', 'object': 'red_bin_1'},
                {'skill': 'pick', 'object': 'red_bin_1'},
            ],
            'object_not_graspable',
        ),
    )
    for subtype, raw_actions, expected_reason in cases:
        spec = generate_scenario(
            CATEGORY_SAFE_REJECTION,
            seed=30,
            subtype=subtype,
        )
        graph = SceneGraph.build_from_environment(
            WorldModel(spec).environment_payload()
        )
        actions = parse_actions(raw_actions)
        _, objects = create_direct_scene(spec)
        try:
            before = {
                name: (
                    p.getBasePositionAndOrientation(object_spec.body_id)[0]
                    if object_spec.body_id >= 0 else None
                )
                for name, object_spec in objects.items()
            }
            report = validate_actions(actions, graph)
            assert report.ok is False
            assert report.reason_code == expected_reason
            after = {
                name: (
                    p.getBasePositionAndOrientation(object_spec.body_id)[0]
                    if object_spec.body_id >= 0 else None
                )
                for name, object_spec in objects.items()
            }
            assert after == before
        finally:
            p.disconnect()


def test_target_pick_plan_executes_in_parameterized_direct_scene():
    spec = generate_scenario(CATEGORY_TARGET_PICK, seed=13)
    target = spec.goal.params['target_id']
    container = spec.goal.params['container_id']
    try:
        objects = run_parameterized_actions(
            spec,
            relocation_actions(target, container),
        )
        position = p.getBasePositionAndOrientation(
            objects[target].body_id
        )[0]
        center = objects[container].position
        assert abs(position[0] - center[0]) < 0.10
        assert abs(position[1] - center[1]) < 0.07
    finally:
        p.disconnect()


def test_sequential_plan_executes_in_parameterized_direct_scene():
    spec = generate_scenario(CATEGORY_SEQUENTIAL, seed=17)
    raw_actions = (
        relocation_actions('cup', 'basket')
        + relocation_actions(
            'apple',
            spec.goal.params['second']['container_id'],
        )
    )
    try:
        objects = run_parameterized_actions(spec, raw_actions)
        for object_id, container_id in (
            ('cup', 'basket'),
            (
                'apple',
                spec.goal.params['second']['container_id'],
            ),
        ):
            position = p.getBasePositionAndOrientation(
                objects[object_id].body_id
            )[0]
            center = objects[container_id].position
            assert abs(position[0] - center[0]) < 0.10
            assert abs(position[1] - center[1]) < 0.07
    finally:
        p.disconnect()


def test_classification_plan_executes_in_parameterized_direct_scene():
    spec = generate_scenario(CATEGORY_CLASSIFICATION, seed=11)
    raw_actions = []
    placements = []
    for pair in spec.goal.params['pairs']:
        for block in sorted(
            spec.instances_of('block', pair['color']),
            key=lambda item: item.instance_id,
        ):
            raw_actions.extend(
                relocation_actions(block.instance_id, pair['bin_id'])
            )
            placements.append((block.instance_id, pair['bin_id']))
    try:
        objects = run_parameterized_actions(spec, raw_actions)
        for object_id, container_id in placements:
            position = p.getBasePositionAndOrientation(
                objects[object_id].body_id
            )[0]
            center = objects[container_id].position
            assert abs(position[0] - center[0]) < 0.10
            assert abs(position[1] - center[1]) < 0.07
    finally:
        p.disconnect()


def test_world_event_rejects_unknown_objects():
    _, objects = create_direct_scene(
        generate_scenario(CATEGORY_RECOVERY, seed=7)
    )
    try:
        acknowledgement = pybullet_robot.apply_world_event(
            objects,
            {
                'revision': 4,
                'event': 'object_slide',
                'target': 'missing_object',
                'new_position': [0.5, 0.0, 0.05],
            },
        )
        assert acknowledgement == {
            'revision': 4,
            'applied': False,
            'reason': 'unknown_target',
        }
    finally:
        p.disconnect()
