"""Tests for the seeded M5 scenario generators."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scenarios.generator import (  # noqa: E402
    GENERATOR_VERSION,
    generate,
)
from agent_robot.scenarios.spec import (  # noqa: E402
    CATEGORY_CLASSIFICATION,
    CATEGORY_RECOVERY,
    CATEGORY_SAFE_REJECTION,
    CATEGORY_SEQUENTIAL,
    CATEGORY_TARGET_PICK,
    REJECT_MISSING_TARGET,
    REJECT_NOT_GRASPABLE,
    REJECT_WRONG_CONTAINER,
    SCENARIO_CATEGORIES,
    validate,
)

ALL_CATEGORIES = list(SCENARIO_CATEGORIES)


@pytest.mark.parametrize('category', ALL_CATEGORIES)
@pytest.mark.parametrize('seed', range(10))
def test_generated_specs_validate(category, seed):
    spec = generate(category, seed)
    validate(spec)
    assert spec.seed == seed
    assert spec.scenario_id == '{}_seed{}'.format(category, seed)


def test_determinism_same_seed_equal():
    for category in ALL_CATEGORIES:
        first = generate(category, 7)
        second = generate(category, 7)
        assert first.to_dict() == second.to_dict()


def test_different_seeds_produce_different_scenes():
    signatures = set()
    for seed in range(20):
        spec = generate(CATEGORY_CLASSIFICATION, seed)
        signature = (
            tuple(
                (obj.instance_id, obj.position) for obj in spec.objects
            ),
            tuple(
                tuple(sorted(obj.color for obj in spec.objects)),
            ),
            len(spec.objects),
        )
        signatures.add(signature)
    assert len(signatures) > 15


def test_seed_isolation_from_global_random(monkeypatched_random):
    first = generate(CATEGORY_TARGET_PICK, 42)
    second = generate(CATEGORY_TARGET_PICK, 42)
    assert first.to_dict() == second.to_dict()


@pytest.fixture
def monkeypatched_random(monkeypatch):
    import random

    def boom(*args, **kwargs):
        raise AssertionError('generator must not touch global random')

    monkeypatch.setattr(random, 'random', boom)
    monkeypatch.setattr(random, 'uniform', boom)
    monkeypatch.setattr(random, 'randint', boom)
    monkeypatch.setattr(random, 'choice', boom)
    monkeypatch.setattr(random, 'shuffle', boom)
    return monkeypatch


def test_classification_structure_and_command():
    spec = generate(CATEGORY_CLASSIFICATION, 1)
    colors = {
        obj.color for obj in spec.objects if obj.object_type == 'block'
    }
    assert 'red' in colors
    assert len(colors) >= 2

    pairs = spec.goal.params['pairs']
    pair_colors = {pair['color'] for pair in pairs}
    assert pair_colors == colors
    for pair in pairs:
        assert spec.object_by_id(pair['bin_id']).color == pair['color']

    assert '把红色积木放进红色箱子' in spec.command
    assert spec.command.endswith('。')


def test_target_pick_goal_objects_exist_and_command_matches():
    for seed in range(10):
        spec = generate(CATEGORY_TARGET_PICK, seed)
        target_id = spec.goal.params['target_id']
        container_id = spec.goal.params['container_id']
        target = spec.object_by_id(target_id)
        container = spec.object_by_id(container_id)
        assert target is not None and target.graspable
        assert container is not None and container.receptacle
        assert len(spec.objects) >= 4


def test_sequential_command_order_and_goal():
    spec = generate(CATEGORY_SEQUENTIAL, 3)
    assert '先把杯子放进篮子' in spec.command
    assert '再把苹果放进' in spec.command
    first = spec.goal.params['first']
    second = spec.goal.params['second']
    assert first['object_id'] == 'cup'
    assert first['container_id'] == 'basket'
    assert second['object_id'] == 'apple'
    assert spec.object_by_id(second['container_id']).object_type == 'bin'


def test_recovery_has_single_slide_after_first_action():
    for seed in range(10):
        spec = generate(CATEGORY_RECOVERY, seed)
        assert len(spec.events) == 1
        event = spec.events[0]
        assert event.after_action_index == 1
        assert event.target == spec.goal.params['target_id']
        assert event.new_position is not None
        target = spec.object_by_id(event.target)
        assert event.new_position != target.position


@pytest.mark.parametrize(
    'subtype',
    [
        REJECT_MISSING_TARGET,
        REJECT_WRONG_CONTAINER,
        REJECT_NOT_GRASPABLE,
    ],
)
def test_safe_rejection_subtypes(subtype):
    spec = generate(
        CATEGORY_SAFE_REJECTION,
        5,
        subtype=subtype,
    )
    assert spec.goal.params['subtype'] == subtype
    validate(spec)


def test_safe_rejection_missing_target_color_absent():
    spec = generate(
        CATEGORY_SAFE_REJECTION,
        1,
        subtype=REJECT_MISSING_TARGET,
    )
    assert '黄色积木' in spec.command
    assert not spec.instances_of('block', 'yellow')


def test_safe_rejection_wrong_container_is_non_receptacle():
    spec = generate(
        CATEGORY_SAFE_REJECTION,
        1,
        subtype=REJECT_WRONG_CONTAINER,
    )
    container = spec.object_by_id(
        spec.goal.params['container_id']
    )
    assert container.receptacle is None
    assert '杯子' in spec.command


def test_safe_rejection_not_graspable_targets_bin():
    spec = generate(
        CATEGORY_SAFE_REJECTION,
        1,
        subtype=REJECT_NOT_GRASPABLE,
    )
    target = spec.object_by_id(spec.goal.params['object_id'])
    assert target.object_type == 'bin'
    assert target.graspable is False
    assert '红' in spec.command and '箱子' in spec.command


def test_subtype_cycles_with_seed_when_omitted():
    subtypes = set()
    for seed in range(6):
        spec = generate(CATEGORY_SAFE_REJECTION, seed)
        subtypes.add(spec.goal.params['subtype'])
    assert subtypes == {
        REJECT_MISSING_TARGET,
        REJECT_WRONG_CONTAINER,
        REJECT_NOT_GRASPABLE,
    }


def test_scenario_id_override():
    spec = generate(
        CATEGORY_RECOVERY, 2, scenario_id='recovery_003'
    )
    assert spec.scenario_id == 'recovery_003'


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        generate('cleanup', 1)


def test_generator_version_constant_present():
    assert GENERATOR_VERSION == 'm5-1'
