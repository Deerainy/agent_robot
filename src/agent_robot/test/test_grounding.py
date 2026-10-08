"""Tests for M5 dual-channel grounding (resolve vs query)."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.plan_validator import (  # noqa: E402
    STAGE_GROUNDING,
    evaluate_generated_plan,
)
from agent_robot.scene_graph import (  # noqa: E402
    OBJECT_TYPES,
    AmbiguousGroundingError,
    GroundingError,
    ObjectNode,
    SceneGraph,
    SkillAction,
    ground_action,
    instance_type_and_color,
    query_instances,
    resolve_instance,
)

resolve_type_color = instance_type_and_color


def make_graph(names):
    graph = SceneGraph(source='test')
    for name in names:
        identity = resolve_type_color(name)
        info = OBJECT_TYPES[identity[0]]
        graph.objects[name] = ObjectNode(
            name=name,
            graspable=info.graspable,
            receptacle=info.receptacle,
        )
    return graph


@pytest.fixture
def multi_graph():
    return make_graph(
        [
            'red_block_1',
            'red_block_2',
            'blue_block_1',
            'red_bin_1',
            'apple',
        ]
    )


def test_exact_instance_id_resolves(multi_graph):
    assert resolve_instance('red_block_1', multi_graph) == \
        'red_block_1'
    assert resolve_instance('RED_BLOCK_2', multi_graph) == \
        'red_block_2'


def test_singleton_chinese_alias_resolves(multi_graph):
    assert resolve_instance('苹果', multi_graph) == 'apple'


def test_unique_color_type_resolves(multi_graph):
    assert resolve_instance('蓝色积木', multi_graph) == 'blue_block_1'
    assert resolve_instance('blue block', multi_graph) == \
        'blue_block_1'


def test_ambiguous_reference_lists_candidates(multi_graph):
    with pytest.raises(AmbiguousGroundingError) as excinfo:
        resolve_instance('红色积木', multi_graph)
    assert excinfo.value.candidates == [
        'red_block_1', 'red_block_2'
    ]
    assert 'red_block_1' in str(excinfo.value)
    assert 'red_block_2' in str(excinfo.value)


def test_missing_reference_raises_grounding_error(multi_graph):
    with pytest.raises(GroundingError):
        resolve_instance('黄色积木', multi_graph)
    with pytest.raises(GroundingError):
        resolve_instance('banana', multi_graph)
    with pytest.raises(GroundingError):
        resolve_instance('', multi_graph)


def test_bin_resolves_through_chinese_alias(multi_graph):
    assert resolve_instance('红箱子', multi_graph) == 'red_bin_1'
    assert resolve_instance('red bin', multi_graph) == 'red_bin_1'


def test_query_instances_filters_and_sorts(multi_graph):
    assert query_instances(
        multi_graph, object_type='block', color='red'
    ) == ['red_block_1', 'red_block_2']
    assert query_instances(
        multi_graph, object_type='block'
    ) == ['blue_block_1', 'red_block_1', 'red_block_2']
    assert query_instances(
        multi_graph, color='red'
    ) == ['red_bin_1', 'red_block_1', 'red_block_2']


def test_query_always_returns_list(multi_graph):
    assert isinstance(query_instances(multi_graph), list)
    assert query_instances(
        multi_graph, object_type='block', color='yellow'
    ) == []
    assert query_instances(
        multi_graph, object_type='积木', color='蓝'
    ) == ['blue_block_1']


def test_ground_action_on_graph_resolves_each_param(multi_graph):
    action = SkillAction(
        skill='pick', params={'object': '蓝色积木'}
    )
    grounded = ground_action(action, multi_graph)
    assert grounded.object == 'blue_block_1'


def test_ground_action_ambiguity_propagates(multi_graph):
    action = SkillAction(
        skill='pick', params={'object': '红色积木'}
    )
    with pytest.raises(AmbiguousGroundingError):
        ground_action(action, multi_graph)


def test_plan_validator_ambiguity_detail_has_candidates(multi_graph):
    generated = {
        'feasible': True,
        'actions': [
            {'skill': 'move_to', 'object': '红色积木'},
            {'skill': 'pick', 'object': '红色积木'},
        ],
    }
    evaluation = evaluate_generated_plan(
        generated, env_names=[], graph=multi_graph
    )
    assert not evaluation.accepted
    assert evaluation.error_stage == STAGE_GROUNDING
    assert evaluation.violation_index == 0
    assert 'red_block_1' in evaluation.error_detail
    assert 'red_block_2' in evaluation.error_detail


def test_plan_validator_missing_target_rejected(multi_graph):
    generated = {
        'feasible': True,
        'actions': [
            {'skill': 'move_to', 'object': '黄色积木'},
        ],
    }
    evaluation = evaluate_generated_plan(
        generated, env_names=[], graph=multi_graph
    )
    assert evaluation.error_stage == STAGE_GROUNDING
    assert 'yellow' in evaluation.error_detail
