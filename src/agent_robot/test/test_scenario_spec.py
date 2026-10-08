"""Tests for M5 SceneSpec structures, JSON conversion and validation."""

import copy
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scenarios.spec import (  # noqa: E402
    CATEGORY_CLASSIFICATION,
    CATEGORY_RECOVERY,
    CATEGORY_SAFE_REJECTION,
    CATEGORY_SEQUENTIAL,
    CATEGORY_TARGET_PICK,
    EVENT_OBJECT_SLIDE,
    REJECT_MISSING_TARGET,
    ObjectInstance,
    SceneSpec,
    ScenarioGoal,
    EventScript,
    scene_spec_from_dict,
    validate,
)


def block(instance_id, x, y, color='red'):
    return ObjectInstance(
        instance_id=instance_id,
        object_type='block',
        color=color,
        position=(x, y, 0.05),
    )


def bin_instance(instance_id, x, y, color='red'):
    return ObjectInstance(
        instance_id=instance_id,
        object_type='bin',
        color=color,
        position=(x, y, 0.05),
    )


def apple(x, y):
    return ObjectInstance(
        instance_id='apple',
        object_type='apple',
        color=None,
        position=(x, y, 0.06),
    )


def cup(x, y):
    return ObjectInstance(
        instance_id='cup',
        object_type='cup',
        color=None,
        position=(x, y, 0.06),
    )


def basket(x, y):
    return ObjectInstance(
        instance_id='basket',
        object_type='basket',
        color=None,
        position=(x, y, 0.05),
    )


def make_spec(category, objects, goal_params, events=()):
    return SceneSpec(
        scenario_id='{}_test'.format(category),
        category=category,
        seed=1,
        command='test command',
        objects=tuple(objects),
        events=tuple(events),
        goal=ScenarioGoal(category=category, params=goal_params),
    )


def classification_spec():
    objects = [
        block('red_block_1', 0.40, -0.20),
        block('red_block_2', 0.40, 0.20),
        block('blue_block_1', 0.50, -0.20, color='blue'),
        bin_instance('red_bin_1', 0.72, 0.20, color='red'),
        bin_instance('blue_bin_1', 0.72, -0.20, color='blue'),
    ]
    return make_spec(
        CATEGORY_CLASSIFICATION,
        objects,
        {'pairs': [
            {'color': 'red', 'bin_id': 'red_bin_1'},
            {'color': 'blue', 'bin_id': 'blue_bin_1'},
        ]},
    )


def recovery_spec():
    objects = [
        apple(0.50, -0.20),
        bin_instance('red_bin_1', 0.72, 0.20),
    ]
    events = [
        EventScript(
            event=EVENT_OBJECT_SLIDE,
            target='apple',
            after_action_index=1,
            new_position=(0.45, 0.30, 0.06),
            new_support='table',
        )
    ]
    return make_spec(
        CATEGORY_RECOVERY,
        objects,
        {'target_id': 'apple', 'container_id': 'red_bin_1'},
        events=events,
    )


def test_valid_classification_spec_passes():
    validate(classification_spec())


def test_valid_recovery_spec_passes():
    validate(recovery_spec())


def test_json_round_trip_is_equal():
    spec = recovery_spec()
    restored = scene_spec_from_dict(spec.to_dict())
    assert restored == spec


def test_all_category_goals_validate():
    spec = make_spec(
        CATEGORY_TARGET_PICK,
        [
            block('green_block_1', 0.40, -0.20, color='green'),
            basket(0.72, 0.20),
        ],
        {'target_id': 'green_block_1', 'container_id': 'basket'},
    )
    validate(spec)

    spec = make_spec(
        CATEGORY_SEQUENTIAL,
        [
            cup(0.40, -0.20),
            apple(0.40, 0.20),
            basket(0.72, 0.20),
            bin_instance('red_bin_1', 0.72, -0.20),
        ],
        {
            'first': {'object_id': 'cup', 'container_id': 'basket'},
            'second': {
                'object_id': 'apple',
                'container_id': 'red_bin_1',
            },
        },
    )
    validate(spec)

    spec = make_spec(
        CATEGORY_SAFE_REJECTION,
        [block('red_block_1', 0.40, -0.20)],
        {'subtype': REJECT_MISSING_TARGET},
    )
    validate(spec)


def test_position_outside_envelope_rejected():
    spec = classification_spec()
    bad = SceneSpec(
        scenario_id=spec.scenario_id,
        category=spec.category,
        seed=spec.seed,
        command=spec.command,
        objects=(
            block('red_block_1', 1.20, -0.20),
            block('red_block_2', 0.40, 0.20),
            block('blue_block_1', 0.55, -0.20, color='blue'),
            bin_instance('red_bin_1', 0.72, 0.20),
            bin_instance('blue_bin_1', 0.72, -0.20, color='blue'),
        ),
        events=(),
        goal=spec.goal,
    )
    with pytest.raises(ValueError):
        validate(bad)


def test_overlapping_objects_rejected():
    spec = classification_spec()
    objects = list(spec.objects)
    objects[1] = block('red_block_2', 0.41, -0.19)
    with pytest.raises(ValueError):
        validate(
            SceneSpec(
                scenario_id=spec.scenario_id,
                category=spec.category,
                seed=spec.seed,
                command=spec.command,
                objects=tuple(objects),
                events=(),
                goal=spec.goal,
            )
        )


def test_object_inside_container_rejected():
    with pytest.raises(ValueError):
        validate(
            make_spec(
                CATEGORY_TARGET_PICK,
                [
                    block('red_block_1', 0.72, 0.20),
                    bin_instance('red_bin_1', 0.72, 0.20),
                ],
                {
                    'target_id': 'red_block_1',
                    'container_id': 'red_bin_1',
                },
            )
        )


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        validate(
            make_spec(
                CATEGORY_SAFE_REJECTION,
                [
                    block('red_block_1', 0.40, -0.20),
                    block('red_block_1', 0.55, 0.20),
                ],
                {'subtype': REJECT_MISSING_TARGET},
            )
        )


def test_instance_id_type_mismatch_rejected():
    raw = classification_spec().to_dict()
    raw['objects'][0]['type'] = 'bin'
    with pytest.raises(ValueError):
        scene_spec_from_dict(raw)


def test_instance_id_color_mismatch_rejected():
    raw = classification_spec().to_dict()
    raw['objects'][0]['color'] = 'blue'
    with pytest.raises(ValueError):
        scene_spec_from_dict(raw)


def test_unknown_event_target_rejected():
    spec = recovery_spec()
    events = [
        EventScript(
            event=EVENT_OBJECT_SLIDE,
            target='red_block_9',
            after_action_index=1,
            new_position=(0.45, 0.30, 0.06),
        )
    ]
    with pytest.raises(ValueError):
        validate(
            SceneSpec(
                scenario_id=spec.scenario_id,
                category=spec.category,
                seed=spec.seed,
                command=spec.command,
                objects=spec.objects,
                events=tuple(events),
                goal=spec.goal,
            )
        )


def test_reposition_event_without_position_rejected():
    spec = recovery_spec()
    events = [
        EventScript(
            event=EVENT_OBJECT_SLIDE,
            target='apple',
            after_action_index=1,
        )
    ]
    with pytest.raises(ValueError):
        validate(
            SceneSpec(
                scenario_id=spec.scenario_id,
                category=spec.category,
                seed=spec.seed,
                command=spec.command,
                objects=spec.objects,
                events=tuple(events),
                goal=spec.goal,
            )
        )


def test_recovery_requires_single_slide_after_first_action():
    spec = recovery_spec()
    no_events = SceneSpec(
        scenario_id=spec.scenario_id,
        category=spec.category,
        seed=spec.seed,
        command=spec.command,
        objects=spec.objects,
        events=(),
        goal=spec.goal,
    )
    with pytest.raises(ValueError):
        validate(no_events)

    wrong_index = [
        EventScript(
            event=EVENT_OBJECT_SLIDE,
            target='apple',
            after_action_index=2,
            new_position=(0.45, 0.30, 0.06),
        )
    ]
    with pytest.raises(ValueError):
        validate(
            SceneSpec(
                scenario_id=spec.scenario_id,
                category=spec.category,
                seed=spec.seed,
                command=spec.command,
                objects=spec.objects,
                events=tuple(wrong_index),
                goal=spec.goal,
            )
        )


def test_goal_pair_must_reference_matching_bin():
    raw = classification_spec().to_dict()
    raw['goal']['params']['pairs'][0]['bin_id'] = 'blue_bin_1'
    with pytest.raises(ValueError):
        scene_spec_from_dict(raw)


def test_goal_pair_requires_matching_blocks():
    raw = classification_spec().to_dict()
    for obj in raw['objects']:
        if obj['id'] in ('red_block_1', 'red_block_2'):
            obj['id'] = 'green_{}'.format(obj['id'])
            obj['type'] = 'block'
            obj['color'] = 'green'
    with pytest.raises(ValueError):
        scene_spec_from_dict(raw)


def test_goal_missing_object_reference_rejected():
    raw = recovery_spec().to_dict()
    raw['goal']['params']['target_id'] = 'block_99'
    with pytest.raises(ValueError):
        scene_spec_from_dict(raw)


def test_deepcopy_independence_check():
    spec = recovery_spec()
    data = spec.to_dict()
    data_copy = copy.deepcopy(data)
    assert data == data_copy
