import os
import sys

import pytest

# Allow running `python3 -m pytest test/test_skill_registry.py` from the
# package root without a colcon/ROS install step.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.skill_registry import (  # noqa: E402
    ActionSchemaError,
    SKILL_NAMES,
    SkillAction,
    SkillExecutionError,
    SkillRegistry,
    parse_action,
    parse_actions,
    skill_catalog_text,
)


def test_parse_valid_action_sequence():
    raw_actions = [
        {'skill': 'move_to', 'object': 'apple'},
        {'skill': 'pick', 'object': 'apple'},
        {'skill': 'move_to', 'object': 'basket'},
        {'skill': 'place', 'object': 'apple', 'target': 'basket'},
    ]

    actions = parse_actions(raw_actions)

    assert [action.skill for action in actions] == [
        'move_to', 'pick', 'move_to', 'place'
    ]
    assert actions[1].object == 'apple'
    assert actions[3].target == 'basket'


def test_values_are_stripped():
    action = parse_action({'skill': ' pick ', 'object': ' apple '})

    assert action.skill == 'pick'
    assert action.object == 'apple'


def test_action_to_dict_and_describe():
    pick = parse_action({'skill': 'pick', 'object': 'apple'})
    place = parse_action(
        {'skill': 'place', 'object': 'apple', 'target': 'basket'}
    )

    assert pick.to_dict() == {'skill': 'pick', 'object': 'apple'}
    assert pick.describe() == 'pick(apple)'
    assert place.describe() == 'place(apple -> basket)'


def test_unknown_skill_is_rejected():
    with pytest.raises(ActionSchemaError, match='unknown skill'):
        parse_action({'skill': 'dance', 'object': 'apple'})


def test_internal_gripper_primitives_are_not_valid_actions():
    with pytest.raises(ActionSchemaError, match='unknown skill'):
        parse_action({'skill': 'open_gripper'})

    with pytest.raises(ActionSchemaError, match='unknown skill'):
        parse_action({'skill': 'close_gripper'})


def test_missing_required_object():
    with pytest.raises(ActionSchemaError, match='"object"'):
        parse_action({'skill': 'pick'})


def test_missing_place_target():
    with pytest.raises(ActionSchemaError, match='"target"'):
        parse_action({'skill': 'place', 'object': 'apple'})


def test_empty_parameter_is_rejected():
    with pytest.raises(ActionSchemaError, match='"object"'):
        parse_action({'skill': 'pick', 'object': '   '})


def test_non_string_parameter_is_rejected():
    with pytest.raises(ActionSchemaError, match='"object"'):
        parse_action({'skill': 'pick', 'object': 3})


def test_unexpected_parameter_is_rejected():
    with pytest.raises(ActionSchemaError, match='unknown parameter'):
        parse_action(
            {'skill': 'pick', 'object': 'apple', 'speed': 'fast'}
        )


def test_action_must_be_a_dict():
    with pytest.raises(ActionSchemaError, match='JSON object'):
        parse_action('pick apple', index=0)


def test_actions_must_be_a_non_empty_list():
    with pytest.raises(ActionSchemaError, match='list'):
        parse_actions({'skill': 'pick'})

    with pytest.raises(ActionSchemaError, match='empty'):
        parse_actions([])


def test_error_message_carries_action_index():
    with pytest.raises(ActionSchemaError, match='actions\\[2\\]'):
        parse_actions([
            {'skill': 'move_to', 'object': 'apple'},
            {'skill': 'pick', 'object': 'apple'},
            {'skill': 'pick'},
        ])


def test_registry_dispatches_to_handler():
    seen = []

    def handler(action, context):
        seen.append((action, context))
        return 'ok'

    registry = SkillRegistry()
    registry.register('pick', handler)

    action = parse_action({'skill': 'pick', 'object': 'cup'})
    result = registry.execute(action, context={'holding': None})

    assert result == 'ok'
    assert seen[0][0].object == 'cup'
    assert seen[0][1] == {'holding': None}


def test_registry_unknown_skill_registration_rejected():
    registry = SkillRegistry()

    with pytest.raises(ValueError, match='unknown skill'):
        registry.register('open_gripper', lambda action, ctx: None)


def test_registry_missing_handler_raises_execution_error():
    registry = SkillRegistry()
    action = SkillAction(skill='pick', params={'object': 'apple'})

    with pytest.raises(SkillExecutionError, match='No handler'):
        registry.execute(action)


def test_registry_wraps_handler_exception():
    def broken_handler(action, context):
        raise RuntimeError('grasp slipped')

    registry = SkillRegistry()
    registry.register('pick', broken_handler)

    action = parse_action({'skill': 'pick', 'object': 'apple'})

    with pytest.raises(SkillExecutionError, match='grasp slipped'):
        registry.execute(action)


def test_registry_preserves_execution_error_identity():
    def raising_handler(action, context):
        raise SkillExecutionError('explicit failure')

    registry = SkillRegistry()
    registry.register('pick', raising_handler)

    action = parse_action({'skill': 'pick', 'object': 'apple'})

    with pytest.raises(SkillExecutionError, match='explicit failure'):
        registry.execute(action)


def test_skill_catalog_text_exposes_high_level_skills_only():
    catalog = skill_catalog_text()

    for name in SKILL_NAMES:
        assert name in catalog

    assert 'open_gripper' not in catalog
    assert 'close_gripper' not in catalog
    assert "'target'" in catalog or 'target' in catalog
