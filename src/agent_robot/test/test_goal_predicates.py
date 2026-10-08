"""Tests for M5 symbolic goal predicates."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scene_graph import SceneGraph  # noqa: E402
from agent_robot.scenarios.goals import evaluate_goal  # noqa: E402
from agent_robot.scenarios.spec import (  # noqa: E402
    CATEGORY_CLASSIFICATION,
    CATEGORY_RECOVERY,
    CATEGORY_SAFE_REJECTION,
    CATEGORY_SEQUENTIAL,
    CATEGORY_TARGET_PICK,
    ScenarioGoal,
)


def graph_with(objects, supports):
    graph = SceneGraph(source='test')
    graph.objects = objects
    graph.supports = supports
    return graph


def object_node(color=None):
    return {
        'graspable': True,
        'receptacle': None,
        'color': color,
        'visible': True,
    }


def block_node(color):
    return {
        'graspable': True,
        'receptacle': None,
        'color': color,
        'visible': True,
        'type': 'block',
    }


def action_event(status, skill, object_id, target=None, index=1):
    return {
        'status': status,
        'step_index': index,
        'action': {
            'skill': skill,
            'params': {'object': object_id, 'target': target},
        },
    }


def test_classification_requires_all_same_color_blocks_in_matching_bin():
    goal = ScenarioGoal(
        category=CATEGORY_CLASSIFICATION,
        params={'pairs': [{'color': 'red', 'bin_id': 'red_bin_1'}]},
    )
    graph = graph_with(
        {
            'red_block_1': block_node('red'),
            'red_block_2': block_node('red'),
            'blue_block_1': block_node('blue'),
        },
        {
            'red_block_1': ('in', 'red_bin_1'),
            'red_block_2': ('in', 'red_bin_1'),
            'blue_block_1': ('on', 'table'),
        },
    )
    assert evaluate_goal(goal, graph, {})['success'] is True

    graph.supports['red_block_2'] = ('in', 'blue_bin_1')
    assert evaluate_goal(goal, graph, {})['success'] is False


def test_classification_rejects_wrong_color_inside_bin():
    goal = {
        'category': CATEGORY_CLASSIFICATION,
        'params': {'pairs': [{'color': 'red', 'bin_id': 'red_bin_1'}]},
    }
    graph = graph_with(
        {
            'red_block_1': block_node('red'),
            'blue_block_1': block_node('blue'),
        },
        {
            'red_block_1': ('in', 'red_bin_1'),
            'blue_block_1': ('in', 'red_bin_1'),
        },
    )
    result = evaluate_goal(goal, graph, {})
    assert result['success'] is False
    assert result['details']['failures'][0]['blocks_in_bin'] == [
        'blue_block_1',
        'red_block_1',
    ]


def test_target_pick_requires_target_and_no_extra_block_in_container():
    goal = ScenarioGoal(
        category=CATEGORY_TARGET_PICK,
        params={'target_id': 'red_block_1', 'container_id': 'basket'},
    )
    graph = graph_with(
        {
            'red_block_1': block_node('red'),
            'blue_block_1': block_node('blue'),
        },
        {'red_block_1': ('in', 'basket')},
    )
    assert evaluate_goal(goal, graph, {})['success'] is True
    graph.supports['blue_block_1'] = ('in', 'basket')
    result = evaluate_goal(goal, graph, {})
    assert result['success'] is False
    assert result['details']['extra_blocks'] == ['blue_block_1']


def test_sequential_checks_completed_place_before_second_pick():
    goal = ScenarioGoal(
        category=CATEGORY_SEQUENTIAL,
        params={
            'first': {'object_id': 'cup', 'container_id': 'basket'},
            'second': {'object_id': 'apple', 'container_id': 'red_bin_1'},
        },
    )
    graph = graph_with(
        {'cup': object_node(), 'apple': object_node()},
        {'cup': ('in', 'basket'), 'apple': ('in', 'red_bin_1')},
    )
    trace = {
        'events': [
            action_event('action_completed', 'move_to', 'cup'),
            action_event('action_completed', 'pick', 'cup'),
            action_event('action_completed', 'move_to', 'basket'),
            action_event('action_completed', 'place', 'cup', 'basket'),
            action_event('action_completed', 'pick', 'apple'),
        ]
    }
    assert evaluate_goal(goal, graph, trace)['success'] is True

    trace['events'].reverse()
    result = evaluate_goal(goal, graph, trace)
    assert result['success'] is False
    assert result['details']['order_ok'] is False


def test_recovery_requires_intercepted_pick_failure_then_success():
    goal = ScenarioGoal(
        category=CATEGORY_RECOVERY,
        params={'target_id': 'apple', 'container_id': 'red_bin_1'},
    )
    graph = graph_with(
        {'apple': object_node()},
        {'apple': ('in', 'red_bin_1')},
    )
    trace = {
        'events': [
            action_event('failed', 'pick', 'apple'),
            {'status': 'succeeded'},
        ]
    }
    assert evaluate_goal(goal, graph, trace)['success'] is True

    trace['events'] = [
        action_event('action_started', 'pick', 'apple'),
        action_event('failed', 'pick', 'apple'),
        {'status': 'succeeded'},
    ]
    assert evaluate_goal(goal, graph, trace)['success'] is False


def test_safe_rejection_requires_zero_started_actions_and_reports_subtype():
    goal = ScenarioGoal(
        category=CATEGORY_SAFE_REJECTION,
        params={'subtype': 'missing_target'},
    )
    trace = {'outcome': 'rejected', 'events': [{'status': 'rejected'}]}
    result = evaluate_goal(goal, {}, trace)
    assert result['success'] is True
    assert result['details']['rejection_subtype'] == 'missing_target'

    trace['events'].insert(
        0, action_event('action_started', 'move_to', 'apple')
    )
    assert evaluate_goal(goal, {}, trace)['success'] is False


def test_evaluator_accepts_serialized_scene_graph():
    goal = ScenarioGoal(
        category=CATEGORY_TARGET_PICK,
        params={'target_id': 'apple', 'container_id': 'basket'},
    )
    graph = {
        'relations': [['in', 'apple', 'basket']],
        'objects': [
            {'name': 'apple', 'type': 'apple'},
            {'name': 'basket', 'type': 'basket'},
        ],
    }
    assert evaluate_goal(goal, graph, {})['success'] is True
