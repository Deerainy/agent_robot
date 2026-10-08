"""Tests for the M5 SceneGraph observation lifecycle."""

import copy
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scene_graph import (  # noqa: E402
    POSITION_EPSILON,
    SceneGraph,
    SkillAction,
)


def frame(revision, objects, scenario_id='classification_001'):
    return {
        'scenario_id': scenario_id,
        'world_revision': revision,
        'objects': objects,
    }


def block(instance_id, x, y, color, location='table', z=0.025,
          visible=True):
    return {
        'id': instance_id,
        'type': 'block',
        'color': color,
        'position': [x, y, z],
        'location': location,
        'visible': visible,
    }


def bin_obj(instance_id, x, y, color):
    return {
        'id': instance_id,
        'type': 'bin',
        'color': color,
        'position': [x, y, 0.05],
        'location': 'table',
    }


def apple(x=0.50, y=-0.20, location='table'):
    return {
        'id': 'apple',
        'name': 'apple',
        'type': 'apple',
        'position': [x, y, 0.06],
        'location': location,
    }


INITIAL_OBJECTS = [
    block('red_block_1', 0.40, -0.20, 'red'),
    block('red_block_2', 0.40, 0.10, 'red'),
    block('blue_block_1', 0.50, -0.20, 'blue'),
    bin_obj('red_bin_1', 0.72, 0.27, 'red'),
    apple(),
]


@pytest.fixture
def graph():
    scene = SceneGraph(source='test')
    scene.initialize_from_environment(
        frame(0, INITIAL_OBJECTS)
    )
    return scene


def test_initialize_recognizes_instances_and_metadata(graph):
    assert graph.scenario_id == 'classification_001'
    assert graph.world_revision == 0
    node = graph.objects['red_block_1']
    assert node.color == 'red'
    assert node.position == (0.40, -0.20, 0.025)
    assert node.graspable is True
    assert graph.objects['red_bin_1'].graspable is False
    assert graph.supports['apple'] == ('on', 'table')


def test_block_inside_bin_becomes_in_relation(graph):
    inside = [
        bin_obj('red_bin_1', 0.72, 0.27, 'red'),
        block(
            'red_block_1', 0.72, 0.27, 'red',
            location='red_bin_1'
        ),
    ]
    scene = SceneGraph(source='test')
    scene.initialize_from_environment(frame(0, inside))
    assert scene.supports['red_block_1'] == ('in', 'red_bin_1')
    assert 'red_bin_1' not in scene.surfaces


def test_update_never_clears_holding(graph):
    graph.holding = 'red_block_1'
    graph.supports.pop('red_block_1', None)
    original_position = graph.objects['red_block_1'].position

    moved_frame = frame(
        1,
        [
            block(
                'red_block_1', 0.72, 0.27, 'red',
                location='red_bin_1'
            ),
            block('red_block_2', 0.40, 0.10, 'red'),
            block('blue_block_1', 0.50, -0.20, 'blue'),
            bin_obj('red_bin_1', 0.72, 0.27, 'red'),
        ],
    )
    assert graph.update_observation(moved_frame) is True
    assert graph.holding == 'red_block_1'
    assert graph.objects['red_block_1'].position == \
        original_position
    assert 'red_block_1' not in graph.supports
    assert any(
        'held_object_external_update:red_block_1' in warning
        for warning in graph.warnings
    )
    assert graph.world_revision == 1


def test_robot_at_invalidated_when_target_moves_far(graph):
    graph.apply(
        SkillAction(skill='move_to', params={'object': 'apple'})
    )
    assert graph.robot_at == 'apple'

    moved = frame(
        1,
        INITIAL_OBJECTS[:-1] + [apple(0.50, 0.25)]
    )
    assert graph.update_observation(moved) is True
    assert graph.robot_at is None


def test_robot_at_survives_unrelated_object_moving(graph):
    graph.apply(
        SkillAction(skill='move_to', params={'object': 'apple'})
    )

    moved = frame(
        1,
        [
            block('red_block_1', 0.42, -0.18, 'red'),
            block('red_block_2', 0.40, 0.10, 'red'),
            block('blue_block_1', 0.50, -0.20, 'blue'),
            bin_obj('red_bin_1', 0.72, 0.27, 'red'),
            apple(),
        ],
    )
    assert graph.update_observation(moved) is True
    assert graph.robot_at == 'apple'


def test_robot_at_survives_position_jitter(graph):
    graph.apply(
        SkillAction(skill='move_to', params={'object': 'apple'})
    )
    jitter = POSITION_EPSILON * 0.5
    moved = frame(
        1,
        INITIAL_OBJECTS[:-1]
        + [apple(0.50 + jitter, -0.20 + jitter)]
    )
    assert graph.update_observation(moved) is True
    assert graph.robot_at == 'apple'


def test_stale_and_duplicate_revisions_ignored(graph):
    newer = frame(
        2,
        [
            block('red_block_1', 0.42, -0.20, 'red'),
            block('red_block_2', 0.40, 0.10, 'red'),
            block('blue_block_1', 0.50, -0.20, 'blue'),
            bin_obj('red_bin_1', 0.72, 0.27, 'red'),
            apple(),
        ],
    )
    assert graph.update_observation(newer) is True
    assert graph.world_revision == 2
    assert graph.objects['red_block_1'].position[0] == 0.42

    assert graph.update_observation(newer) is False
    older = copy.deepcopy(newer)
    older['world_revision'] = 1
    assert graph.update_observation(older) is False
    assert graph.world_revision == 2
    assert graph.objects['red_block_1'].position[0] == 0.42


def test_disappearing_object_marked_hidden_then_reshown(graph):
    missing = frame(1, INITIAL_OBJECTS[:-1])
    assert graph.update_observation(missing) is True
    assert graph.objects['apple'].visible is False

    restored = frame(2, INITIAL_OBJECTS)
    assert graph.update_observation(restored) is True
    assert graph.objects['apple'].visible is True


def test_hidden_object_blocks_move_to_and_pick(graph):
    missing = frame(1, INITIAL_OBJECTS[:-1])
    graph.update_observation(missing)

    move = SkillAction(
        skill='move_to', params={'object': 'apple'}
    )
    assert graph.check_preconditions(move) is not None

    graph.robot_at = 'apple'
    pick = SkillAction(skill='pick', params={'object': 'apple'})
    reason, _ = graph.check_preconditions(pick)
    assert reason == 'object_not_in_scene'


def test_legacy_revisionless_frame_still_applies(graph):
    legacy = {'objects': [apple(0.50, 0.25)]}
    assert graph.update_observation(legacy) is True
    assert graph.world_revision == 0


def test_dict_round_trip_carries_m5_fields(graph):
    data = graph.to_dict()
    restored = SceneGraph.from_dict(copy.deepcopy(data))
    assert restored.scenario_id == 'classification_001'
    assert restored.world_revision == 0
    assert restored.objects['red_block_1'].color == 'red'
    assert restored.objects['red_block_1'].position == \
        (0.40, -0.20, 0.025)
    assert restored.supports['red_block_1'] == ('on', 'table')


def test_legacy_dict_without_m5_fields_readable():
    legacy = {
        'schema_version': '1.0',
        'source': 'perception',
        'warnings': [],
        'robot': {'holding': None, 'at': None},
        'objects': [
            {
                'name': 'apple',
                'graspable': True,
                'receptacle': None,
                'support': {'relation': 'on', 'target': 'table'},
            },
            {
                'name': 'basket',
                'graspable': False,
                'receptacle': 'container',
                'support': None,
            },
        ],
        'surfaces': [
            {'name': 'table', 'receptacle': 'surface', 'known': True}
        ],
        'relations': [['on', 'apple', 'table']],
    }
    restored = SceneGraph.from_dict(legacy)
    assert restored.world_revision == 0
    assert restored.scenario_id is None
    assert restored.objects['apple'].visible is True
    assert restored.supports['apple'] == ('on', 'table')


def test_prompt_text_mentions_color_and_type(graph):
    text = graph.to_prompt_text()
    assert 'red_block_1' in text
    assert '红色' in text
    assert '积木' in text
