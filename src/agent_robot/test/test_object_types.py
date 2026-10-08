"""Tests for the M5 object type table and instance identity helpers."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scene_graph import (  # noqa: E402
    CHINESE_COLOR_WORDS,
    MULTI_INSTANCE_TYPES,
    OBJECT_CATALOG,
    OBJECT_TYPES,
    SCENE_COLORS,
    build_instance_id,
    parse_instance_id,
)


def test_legacy_singleton_types_unchanged():
    assert OBJECT_TYPES['apple'].graspable is True
    assert OBJECT_TYPES['apple'].receptacle is None
    assert OBJECT_TYPES['cup'].graspable is True
    assert OBJECT_TYPES['basket'].graspable is False
    assert OBJECT_TYPES['basket'].receptacle == 'container'
    assert set(OBJECT_CATALOG) == {'apple', 'cup', 'basket'}


def test_new_multi_instance_types():
    assert OBJECT_TYPES['block'].graspable is True
    assert OBJECT_TYPES['block'].receptacle is None
    assert OBJECT_TYPES['bin'].graspable is False
    assert OBJECT_TYPES['bin'].receptacle == 'container'
    assert MULTI_INSTANCE_TYPES == ('apple', 'block', 'bin')


def test_scene_color_palettes():
    assert SCENE_COLORS == ('red', 'blue', 'green', 'yellow')
    assert set(CHINESE_COLOR_WORDS.values()) == set(SCENE_COLORS)
    assert CHINESE_COLOR_WORDS['红'] == 'red'


def test_instance_id_round_trip():
    instance_id = build_instance_id('block', 'red', 2)
    assert instance_id == 'red_block_2'

    parsed = parse_instance_id(instance_id)
    assert parsed is not None
    assert parsed.object_type == 'block'
    assert parsed.color == 'red'
    assert parsed.index == 2


def test_build_instance_id_accepts_apple_instances():
    assert build_instance_id('apple', 'red', 1) == 'red_apple_1'


def test_build_instance_id_rejects_unknown_color():
    with pytest.raises(ValueError):
        build_instance_id('block', 'purple', 1)


def test_build_instance_id_rejects_bad_index():
    with pytest.raises(ValueError):
        build_instance_id('block', 'red', 0)


@pytest.mark.parametrize(
    'name',
    [
        'apple',
        'basket',
        'red_apple',
        'red_block',
        'block_1',
        'red_block_a',
        'purple_block_1',
        '',
        'RED_BLOCK_1',
    ],
)
def test_parse_instance_id_rejects_non_instance_names(name):
    assert parse_instance_id(name) is None


def test_parse_instance_id_handles_non_string():
    assert parse_instance_id(None) is None


def test_parse_instance_id_supports_colored_apple_instances():
    parsed = parse_instance_id('red_apple_1')
    assert parsed is not None
    assert parsed.object_type == 'apple'
    assert parsed.color == 'red'
    assert parsed.index == 1
