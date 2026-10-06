import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.plan_validator import (  # noqa: E402
    STAGE_GROUNDING,
    STAGE_SCHEMA,
    STAGE_VALIDATION,
    build_validation_feedback,
    evaluate_generated_plan,
)
from agent_robot.scene_graph import SceneGraph  # noqa: E402


def default_graph():
    return SceneGraph.build_default()


def perceived_names():
    return ['red_apple', 'basket', 'cup']


def valid_doc():
    return {
        'feasible': True,
        'reason': '',
        'actions': [
            {'skill': 'move_to', 'object': 'red_apple'},
            {'skill': 'pick', 'object': 'red_apple'},
            {'skill': 'move_to', 'object': 'basket'},
            {'skill': 'place', 'object': 'red_apple', 'target': 'basket'},
        ],
    }


def test_valid_plan_accepted_with_canonical_grounding():
    result = evaluate_generated_plan(
        valid_doc(), perceived_names(), default_graph()
    )

    assert result.accepted is True
    assert result.status == 'planned'
    assert result.actions == [
        {'skill': 'move_to', 'object': 'apple'},
        {'skill': 'pick', 'object': 'apple'},
        {'skill': 'move_to', 'object': 'basket'},
        {'skill': 'place', 'object': 'apple', 'target': 'basket'},
    ]


def test_infeasible_empty_actions_is_clean_rejection():
    result = evaluate_generated_plan(
        {'feasible': False, 'reason': '缺少目标', 'actions': []},
        perceived_names(),
        default_graph()
    )

    assert result.feasible is False
    assert result.accepted is False
    assert result.repairable is False
    assert result.status == 'rejected'


def test_infeasible_with_actions_is_schema_error():
    doc = {'feasible': False, 'reason': '', 'actions': [
        {'skill': 'pick', 'object': 'apple'}
    ]}

    result = evaluate_generated_plan(
        doc, perceived_names(), default_graph()
    )

    assert result.error_stage == STAGE_SCHEMA
    assert result.repairable is True


def test_unknown_skill_is_schema_error():
    doc = {
        'feasible': True,
        'actions': [{'skill': 'open_gripper'}],
    }

    result = evaluate_generated_plan(
        doc, perceived_names(), default_graph()
    )

    assert result.error_stage == STAGE_SCHEMA
    assert 'open_gripper' in result.error_detail


def test_bare_pick_is_validation_error_robot_not_at():
    doc = {
        'feasible': True,
        'actions': [{'skill': 'pick', 'object': 'apple'}],
    }

    result = evaluate_generated_plan(
        doc, perceived_names(), default_graph()
    )

    assert result.error_stage == STAGE_VALIDATION
    assert result.error_code == 'robot_not_at_object'
    assert result.violation_index == 0
    assert result.violation_action == {'skill': 'pick', 'object': 'apple'}
    assert result.repairable is True


def test_pick_basket_is_validation_error_not_graspable():
    doc = {
        'feasible': True,
        'actions': [
            {'skill': 'move_to', 'object': 'basket'},
            {'skill': 'pick', 'object': 'basket'},
        ],
    }

    result = evaluate_generated_plan(
        doc, perceived_names(), default_graph()
    )

    assert result.error_stage == STAGE_VALIDATION
    assert result.error_code == 'object_not_graspable'


def test_unknown_object_is_grounding_error():
    doc = {
        'feasible': True,
        'actions': [
            {'skill': 'move_to', 'object': 'banana'},
            {'skill': 'pick', 'object': 'banana'},
        ],
    }

    result = evaluate_generated_plan(
        doc, perceived_names(), default_graph()
    )

    assert result.error_stage == STAGE_GROUNDING
    assert result.violation_index == 0
    assert result.violation_action == {
        'skill': 'move_to', 'object': 'banana'
    }


def test_validation_skipped_when_graph_unavailable():
    # No graph and no environment: canonical self-fallback still grounds the
    # known objects; sequence validation is deliberately skipped.
    doc = {
        'feasible': True,
        'actions': [{'skill': 'pick', 'object': 'apple'}],
    }

    result = evaluate_generated_plan(
        doc, ['apple'], None
    )

    assert result.accepted is True
    assert result.actions[0] == {'skill': 'pick', 'object': 'apple'}


def test_non_object_document_is_schema_error():
    result = evaluate_generated_plan(
        'not json', perceived_names(), default_graph()
    )

    assert result.error_stage == STAGE_SCHEMA
    assert result.repairable is True


def test_feedback_text_mentions_violation_and_scene():
    doc = {
        'feasible': True,
        'actions': [{'skill': 'pick', 'object': 'apple'}],
    }
    result = evaluate_generated_plan(
        doc, perceived_names(), default_graph()
    )

    text = build_validation_feedback(result, default_graph())

    assert 'actions[0]' in text
    assert 'robot_not_at_object' in text
    assert 'basket' in text
    assert '严禁原样重试' in text


def test_feedback_for_grounding_error():
    doc = {
        'feasible': True,
        'actions': [
            {'skill': 'move_to', 'object': 'apple'},
            {'skill': 'pick', 'object': 'apple'},
            {'skill': 'move_to', 'object': 'banana'},
        ],
    }
    result = evaluate_generated_plan(
        doc, perceived_names(), default_graph()
    )

    assert result.error_stage == STAGE_GROUNDING
    text = build_validation_feedback(result, default_graph())
    assert '物体无法在当前场景中找到' in text
    assert 'actions[2]' in text
