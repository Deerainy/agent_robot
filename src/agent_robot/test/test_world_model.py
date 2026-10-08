"""Tests for the M5 WorldModel truth state machine."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scenarios.generator import generate_scenario  # noqa: E402
from agent_robot.scenarios.spec import (  # noqa: E402
    CATEGORY_RECOVERY,
    EVENT_GRASP_FAILURE,
    EVENT_OCCLUDE,
    EVENT_REVEAL,
    EventScript,
    SceneSpec,
    ScenarioGoal,
    ObjectInstance,
)
from agent_robot.scenarios.world_model import (  # noqa: E402
    EventPreconditionError,
    WorldModel,
)


def test_recovery_slide_after_move_to():
    spec = generate_scenario(CATEGORY_RECOVERY, seed=7)
    world = WorldModel(spec, physical=False)
    payload0 = world.environment_payload()
    assert payload0['world_revision'] == 0

    target_id = spec.goal.params['target_id']
    target0 = next(
        obj for obj in payload0['objects'] if obj['id'] == target_id
    )
    pos0 = list(target0['position'])

    assert world.on_action_started('move_to', target_id, 1) is None
    result = world.on_action_completed(
        'move_to', target_id, None, 1
    )
    assert result.error is None
    assert world.revision == 1

    payload1 = world.environment_payload()
    target1 = next(
        obj for obj in payload1['objects'] if obj['id'] == target_id
    )
    assert target1['position'] != pos0
    assert len(world.timeline) >= 1

    assert world.on_action_started('pick', target_id, 2) is None
    world.on_action_completed('pick', target_id, None, 2)
    assert world.holding == target_id

    container_id = spec.goal.params['container_id']
    world.on_action_completed('move_to', container_id, None, 3)
    world.on_action_completed(
        'place', target_id, container_id, 4
    )
    assert world.holding is None
    placed = world.instances[target_id]
    assert placed.support == container_id


def test_occlude_reveal_and_empty_hand_guard():
    spec = SceneSpec(
        scenario_id='occlude_test',
        category='target_pick',
        seed=1,
        command='test',
        objects=(
            ObjectInstance(
                instance_id='apple',
                object_type='apple',
                color=None,
                position=(0.5, 0.0, 0.06),
            ),
        ),
        goal=ScenarioGoal(category='target_pick', params={}),
        events=(
            EventScript(
                event=EVENT_OCCLUDE,
                target='apple',
                after_action_index=1,
            ),
            EventScript(
                event=EVENT_REVEAL,
                target='apple',
                after_action_index=2,
            ),
        ),
    )
    world = WorldModel(spec)
    world.on_action_completed('move_to', 'apple', None, 1)
    assert world.revision == 1
    assert not world.instances['apple'].visible

    world.holding = 'apple'
    with pytest.raises(EventPreconditionError):
        world.on_action_completed('move_to', 'apple', None, 2)

    world.holding = None
    world.on_action_completed('move_to', 'apple', None, 2)
    assert world.instances['apple'].visible
    assert world.revision == 2


def test_grasp_failure_inject_once():
    spec = SceneSpec(
        scenario_id='grasp_test',
        category='recovery',
        seed=2,
        command='test',
        objects=(
            ObjectInstance(
                instance_id='apple',
                object_type='apple',
                color=None,
                position=(0.5, 0.0, 0.06),
            ),
        ),
        goal=ScenarioGoal(category='recovery', params={}),
        events=(
            EventScript(
                event=EVENT_GRASP_FAILURE,
                target='apple',
                after_action_index=2,
            ),
        ),
    )
    world = WorldModel(spec)
    inject = world.on_action_started('pick', 'apple', 2)
    assert inject is not None
    assert inject.skill == 'pick'
    assert world.on_action_started('pick', 'apple', 2) is None


def test_physical_mode_pending_until_ack():
    spec = generate_scenario(CATEGORY_RECOVERY, seed=3)
    world = WorldModel(spec, physical=True)
    target_id = spec.goal.params['target_id']

    world.on_action_completed('move_to', target_id, None, 1)
    assert world.revision == 0
    assert world._pending is not None

    commands = world.drain_commands()
    assert len(commands) == 1
    command = commands[0]
    assert command.revision == 1

    assert world.on_event_ack(1, True)
    assert world.revision == 1
    assert world._pending is None

    assert not world.on_event_ack(1, True)


def test_consumed_slide_not_replayed_after_replan():
    spec = generate_scenario(CATEGORY_RECOVERY, seed=9)
    world = WorldModel(spec)
    target_id = spec.goal.params['target_id']
    event = spec.events[0]

    world.on_action_completed('move_to', target_id, None, 1)
    assert world.event_already_consumed(event)

    revision_after_first = world.revision
    world.on_action_completed('move_to', target_id, None, 1)
    assert world.revision == revision_after_first
