import os
import sys

import pytest

# Allow running `python3 -m pytest test/test_scene_graph.py` from the
# package root without a colcon/ROS install step.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scene_graph import (  # noqa: E402
    HAND_OCCUPIED,
    NOT_HOLDING_OBJECT,
    OBJECT_NOT_GRASPABLE,
    OBJECT_NOT_IN_SCENE,
    ROBOT_NOT_AT_OBJECT,
    SCENE_GRAPH_VERSION,
    SOURCE_DEFAULT,
    SOURCE_PERCEPTION,
    TARGET_NOT_RECEPTACLE,
    GroundingError,
    SceneGraph,
    ground_action,
    ground_name,
    validate_actions,
)
from agent_robot.skill_registry import SkillAction  # noqa: E402


def move_to(name):
    return SkillAction(skill='move_to', params={'object': name})


def pick(name):
    return SkillAction(skill='pick', params={'object': name})


def place(name, target):
    return SkillAction(
        skill='place', params={'object': name, 'target': target}
    )


PICK_PLACE_ACTIONS = [
    move_to('apple'),
    pick('apple'),
    move_to('basket'),
    place('apple', 'basket'),
]


# ---------------------------------------------------------------------------
# Grounding regression (moved from task_executor in M2)
# ---------------------------------------------------------------------------

def test_ground_name_known_aliases():
    assert ground_name('苹果', []) == 'apple'
    assert ground_name('篮子', []) == 'basket'
    assert ground_name('水杯', []) == 'cup'


def test_ground_name_strips_color_words():
    assert ground_name('red_apple', ['red_apple']) == 'apple'
    assert ground_name('BLUE CUP', ['blue cup']) == 'cup'


def test_ground_name_exact_environment_match():
    assert ground_name('basket', ['basket']) == 'basket'


def test_ground_name_unknown_object_returns_none():
    assert ground_name('book', ['book']) is None
    assert ground_name('', []) is None
    assert ground_name(None, []) is None


def test_ground_action_raises_on_unknown_parameter():
    action = SkillAction(skill='pick', params={'object': 'banana'})

    with pytest.raises(GroundingError):
        ground_action(action, ['banana'])


# ---------------------------------------------------------------------------
# Default / perceived graph construction
# ---------------------------------------------------------------------------

def test_build_default_scene():
    graph = SceneGraph.build_default()

    assert graph.source == SOURCE_DEFAULT
    assert graph.holding is None
    assert graph.robot_at is None
    assert set(graph.objects) == {'apple', 'cup', 'basket'}
    assert graph.objects['apple'].graspable is True
    assert graph.objects['basket'].graspable is False
    assert graph.objects['basket'].receptacle == 'container'
    assert graph.supports['apple'] == ('on', 'table')
    assert graph.supports['basket'] == ('on', 'table')
    assert graph.surfaces['table'].receptacle == 'surface'
    assert graph.warnings == []


def test_build_from_environment_grounds_and_keeps_known_locations():
    environment = {
        'objects': [
            {'name': 'red_apple', 'color': 'red', 'location': 'desk'},
            {'name': 'basket', 'color': 'brown', 'location': 'floor'},
        ]
    }

    graph = SceneGraph.build_from_environment(environment)

    assert graph.source == SOURCE_PERCEPTION
    assert 'apple' in graph.objects
    assert graph.supports['apple'] == ('on', 'desk')
    assert graph.supports['basket'] == ('on', 'floor')
    assert graph.surfaces['desk'].receptacle == 'surface'
    assert graph.warnings == []


def test_build_from_environment_unknown_location_is_kept_not_table():
    environment = {
        'objects': [
            {'name': 'apple', 'location': 'shelf'},
        ]
    }

    graph = SceneGraph.build_from_environment(environment)

    assert graph.supports['apple'] == ('on', 'shelf')
    assert 'table' not in graph.surfaces
    surface = graph.surfaces['shelf']
    assert surface.receptacle is None
    assert surface.known is False
    assert 'unknown_support:shelf' in graph.warnings


def test_build_from_environment_missing_location_warns_without_support():
    environment = {
        'objects': [
            {'name': 'apple', 'location': ''},
            {'name': 'cup'},
        ]
    }

    graph = SceneGraph.build_from_environment(environment)

    assert 'apple' not in graph.supports
    assert 'cup' not in graph.supports
    assert 'missing_support:apple' in graph.warnings
    assert 'missing_support:cup' in graph.warnings


def test_build_from_environment_ignores_uncatalogued_object():
    environment = {
        'objects': [
            {'name': 'book', 'location': 'table'},
            {'name': 'apple', 'location': 'table'},
        ]
    }

    graph = SceneGraph.build_from_environment(environment)

    assert 'book' not in graph.objects
    assert 'unknown_object:book' in graph.warnings
    assert set(graph.objects) == {'apple'}


def test_clone_isolation():
    graph = SceneGraph.build_default()
    cloned = graph.clone()

    cloned.apply(move_to('apple'))
    cloned.apply(pick('apple'))

    assert graph.robot_at is None
    assert graph.holding is None
    assert graph.supports['apple'] == ('on', 'table')
    assert cloned.robot_at == 'apple'
    assert cloned.holding == 'apple'
    assert 'apple' not in cloned.supports


# ---------------------------------------------------------------------------
# Precondition validation
# ---------------------------------------------------------------------------

def test_valid_pick_place_sequence():
    graph = SceneGraph.build_default()
    report = validate_actions(PICK_PLACE_ACTIONS, graph)

    assert report.ok is True
    assert report.checks == [True, True, True, True]
    assert report.violation is None

    # Original graph stays untouched; the final graph is the prediction.
    assert graph.holding is None
    final = report.final_graph
    assert final.holding is None
    assert final.robot_at == 'basket'
    assert final.supports['apple'] == ('in', 'basket')


def test_pick_without_move_to_violates_robot_not_at():
    graph = SceneGraph.build_default()
    report = validate_actions([pick('apple')], graph)

    assert report.ok is False
    assert report.violation.index == 0
    assert report.violation.reason_code == ROBOT_NOT_AT_OBJECT
    assert graph.robot_at is None


def test_pick_basket_violates_not_graspable():
    graph = SceneGraph.build_default()
    report = validate_actions(
        [move_to('basket'), pick('basket')], graph
    )

    assert report.ok is False
    assert report.violation.index == 1
    assert report.violation.reason_code == OBJECT_NOT_GRASPABLE


def test_second_pick_violates_hand_occupied():
    graph = SceneGraph.build_default()
    actions = [
        move_to('apple'),
        pick('apple'),
        move_to('cup'),
        pick('cup'),
    ]

    report = validate_actions(actions, graph)

    assert report.ok is False
    assert report.violation.index == 3
    assert report.violation.reason_code == HAND_OCCUPIED


def test_place_with_empty_hand_violates_not_holding():
    graph = SceneGraph.build_default()
    actions = [
        move_to('basket'),
        place('apple', 'basket'),
    ]

    report = validate_actions(actions, graph)

    assert report.ok is False
    assert report.violation.index == 1
    assert report.violation.reason_code == NOT_HOLDING_OBJECT


def test_place_wrong_object_violates_not_holding():
    graph = SceneGraph.build_default()
    actions = [
        move_to('apple'),
        pick('apple'),
        move_to('basket'),
        place('cup', 'basket'),
    ]

    report = validate_actions(actions, graph)

    assert report.ok is False
    assert report.violation.index == 3
    assert report.violation.reason_code == NOT_HOLDING_OBJECT


def test_place_onto_object_violates_target_not_receptacle():
    graph = SceneGraph.build_default()
    actions = [
        move_to('apple'),
        pick('apple'),
        move_to('cup'),
        place('apple', 'cup'),
    ]

    report = validate_actions(actions, graph)

    assert report.ok is False
    assert report.violation.index == 3
    assert report.violation.reason_code == TARGET_NOT_RECEPTACLE


def test_place_onto_unknown_surface_violates_not_receptacle():
    graph = SceneGraph.build_from_environment(
        {'objects': [{'name': 'apple', 'location': 'shelf'}]}
    )
    # Manually drive the robot into a holding state at the unknown surface.
    graph.robot_at = 'shelf'
    graph.holding = 'apple'
    graph.supports.pop('apple', None)

    report = validate_actions([place('apple', 'shelf')], graph)

    assert report.ok is False
    assert report.violation.reason_code == TARGET_NOT_RECEPTACLE


def test_place_without_move_to_target_violates_robot_not_at():
    graph = SceneGraph.build_default()
    graph.robot_at = 'apple'
    graph.holding = 'apple'
    graph.supports.pop('apple', None)

    report = validate_actions([place('apple', 'basket')], graph)

    assert report.ok is False
    assert report.violation.reason_code == ROBOT_NOT_AT_OBJECT


def test_move_to_surface_or_unknown_object_rejected():
    graph = SceneGraph.build_default()

    report = validate_actions([move_to('table')], graph)
    assert report.ok is False
    assert report.violation.reason_code == OBJECT_NOT_IN_SCENE

    report = validate_actions([pick('table')], graph)
    assert report.ok is False
    assert report.violation.reason_code == OBJECT_NOT_IN_SCENE

    report = validate_actions(
        [move_to('banana')], graph
    )
    assert report.ok is False
    assert report.violation.reason_code == OBJECT_NOT_IN_SCENE


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_to_dict_contains_protocol_fields():
    graph = SceneGraph.build_default()
    data = graph.to_dict()

    assert data['schema_version'] == SCENE_GRAPH_VERSION
    assert data['source'] == SOURCE_DEFAULT
    assert data['robot'] == {'holding': None, 'at': None}
    names = {obj['name'] for obj in data['objects']}
    assert names == {'apple', 'cup', 'basket'}
    assert ['on', 'apple', 'table'] in data['relations']
    surface_names = {surface['name'] for surface in data['surfaces']}
    assert 'table' in surface_names


def test_from_dict_roundtrip_mid_execution_state():
    graph = SceneGraph.build_default()
    graph.source = 'execution'
    graph.apply(move_to('apple'))
    graph.apply(pick('apple'))
    graph.apply(move_to('basket'))

    restored = SceneGraph.from_dict(graph.to_dict())

    assert restored.source == 'execution'
    assert restored.holding == 'apple'
    assert restored.robot_at == 'basket'
    assert 'apple' not in restored.supports
    assert restored.objects['basket'].receptacle == 'container'

    report = validate_actions([place('apple', 'basket')], restored)
    assert report.ok is True
    assert report.final_graph.supports['apple'] == ('in', 'basket')


def test_from_dict_preserves_unknown_surface_warning():
    graph = SceneGraph.build_from_environment(
        {'objects': [{'name': 'cup', 'location': 'shelf'}]}
    )

    restored = SceneGraph.from_dict(graph.to_dict())

    assert restored.surfaces['shelf'].receptacle is None
    assert restored.supports['cup'] == ('on', 'shelf')
    assert 'unknown_support:shelf' in restored.warnings


def test_prompt_text_mentions_properties_and_robot_state():
    initial = SceneGraph.build_default().to_prompt_text()
    assert 'basket' in initial
    assert '容器' in initial
    assert '手空' in initial

    graph = SceneGraph.build_default()
    graph.apply(move_to('apple'))
    graph.apply(pick('apple'))
    text = graph.to_prompt_text()
    assert '握持 apple' in text
    assert '位于 apple 上方' in text
