"""Image-derived simulation placement regression tests."""

import math
import os
import sys

import cv2
import numpy as np
from types import SimpleNamespace
from unittest import mock

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.perception.coordinate_mapper import (  # noqa: E402
    CoordinateMapper,
    clamp_bbox,
    correct_simulation_placements,
    validate_object_distances,
)
from agent_robot.perception.pipeline import perceive_image  # noqa: E402
from agent_robot.perception import (  # noqa: E402
    pipeline as perception_pipeline,
)
from agent_robot.pybullet_robot import (  # noqa: E402
    build_skill_handlers,
    create_scene,
)
from agent_robot.scenarios.spec import ObjectInstance  # noqa: E402
from agent_robot.skill_registry import SkillAction  # noqa: E402

IMAGE_2 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'images',
    '2.png',
)


def test_bbox_clamps_to_image_and_rejects_non_intersecting_boxes():
    assert clamp_bbox([-5, 2, 110, 90], (100, 80)) == [
        0.0, 2.0, 100.0, 80.0
    ]
    mapper = CoordinateMapper()
    assert mapper.map_bbox([-5, 2, 110, 90], (100, 80), 'cup')
    try:
        mapper.map_bbox([110, 2, 120, 20], (100, 80), 'cup')
    except ValueError as error:
        assert 'does not intersect' in str(error)
    else:
        raise AssertionError('Out-of-image bbox was accepted.')


def test_bad_coordinate_values_are_rejected():
    mapper = CoordinateMapper()
    for pixel in ((float('nan'), 2), (1, float('inf'))):
        try:
            mapper.pixel_to_world(pixel, (100, 80))
        except ValueError as error:
            assert 'finite' in str(error)
        else:
            raise AssertionError('Non-finite pixel was accepted.')


def test_off_image_detection_does_not_discard_valid_scene(tmp_path):
    image_path = str(tmp_path / 'scene.png')
    assert cv2.imwrite(
        image_path,
        np.zeros((80, 100, 3), dtype='uint8'),
    )
    detections = [
        {
            'id': 'red_apple_1',
            'name': 'red_apple',
            'type': 'apple',
            'color': 'red',
            'bbox': [20, 20, 40, 40],
            'confidence': 0.9,
        },
        {
            'id': 'hallucinated_cup',
            'name': 'cup',
            'type': 'cup',
            'color': None,
            'bbox': [120, 20, 150, 50],
            'confidence': 0.3,
        },
    ]
    with mock.patch.object(
        perception_pipeline,
        'detect_image',
        return_value=detections,
    ):
        perception, environment, graph = perceive_image(
            image_path,
            vision_backend='opencv',
        )

    assert perception['objects'] == detections
    assert [item['id'] for item in environment['objects']] == [
        'red_apple_1'
    ]
    assert set(graph.objects) == {'red_apple_1'}
    warning = environment['simulation_placement']['warnings'][0]
    assert 'hallucinated_cup' in warning
    assert 'does not intersect the image' in warning


def test_partially_off_image_detection_is_clamped_for_position(tmp_path):
    image_path = str(tmp_path / 'scene.png')
    assert cv2.imwrite(
        image_path,
        np.zeros((80, 100, 3), dtype='uint8'),
    )
    detections = [{
        'id': 'basket',
        'name': 'basket',
        'type': 'basket',
        'color': 'brown',
        'bbox': [75, 20, 120, 60],
        'confidence': 0.8,
    }]
    with mock.patch.object(
        perception_pipeline,
        'detect_image',
        return_value=detections,
    ):
        perception, environment, graph = perceive_image(
            image_path,
            vision_backend='opencv',
        )

    assert perception['objects'][0]['bbox'] == [75, 20, 120, 60]
    assert environment['objects'][0]['bbox'] == [75, 20, 120, 60]
    assert environment['objects'][0]['position'] == \
        CoordinateMapper().map_bbox([75, 20, 100, 60], (100, 80), 'basket')
    assert 'basket' in graph.objects
    assert environment['simulation_objects'][0]['bbox'] == [
        75.0, 20.0, 100.0, 60.0
    ]


def test_image_indicates_containment_is_preserved_in_simulation_layout():
    result = correct_simulation_placements([
        {
            'id': 'basket',
            'type': 'basket',
            'position': [0.6, 0.0, 0.05],
            'bbox': [20, 20, 100, 100],
        },
        {
            'id': 'cup',
            'type': 'cup',
            'position': [0.6, 0.0, 0.06],
            'bbox': [45, 45, 55, 60],
        },
    ])

    assert result['containment'] == [['cup', 'basket']]
    objects = {item['id']: item for item in result['objects']}
    assert objects['cup']['position'] == [0.6, 0.0, 0.08]
    assert result['warnings'] == []


def test_graspable_objects_keep_clearance_from_receptacles():
    result = correct_simulation_placements([
        {
            'id': 'basket',
            'type': 'basket',
            'position': [0.6, 0.0, 0.05],
        },
        {
            'id': 'cup',
            'type': 'cup',
            'position': [0.6, 0.0, 0.06],
        },
    ])

    objects = {item['id']: item for item in result['objects']}
    distance = math.dist(
        objects['cup']['position'][:2],
        objects['basket']['position'][:2],
    )
    minimum = (
        math.hypot(0.13, 0.10)
        + math.hypot(0.045, 0.045)
        + 0.005
        + 0.07
    )
    assert distance >= minimum
    assert validate_object_distances(result['objects']) == []


def test_image_2_keeps_perception_positions_but_corrects_simulation():
    perception, environment, graph = perceive_image(
        IMAGE_2,
        vision_backend='opencv',
    )
    raw_objects = environment['objects']
    simulation_objects = environment['simulation_objects']
    raw_positions = {
        item['id']: list(item['position']) for item in raw_objects
    }
    raw_boxes = {item['id']: item['bbox'] for item in raw_objects}
    corrected_positions = {
        item['id']: list(item['position']) for item in simulation_objects
    }
    simulation_boxes = {
        item['id']: item['bbox'] for item in simulation_objects
    }

    assert environment['simulation_placement']['warnings'] == [
        'initial placement collision detected'
    ]
    assert raw_positions['cup'] != corrected_positions['cup']
    assert {
        item['name']: item['position'] for item in graph.to_dict()['objects']
    } == {
        item['id']: item['position'] for item in raw_objects
        if item['type'] != 'table'
    }
    assert {
        item['name']: item['bbox'] for item in graph.to_dict()['objects']
    } == {
        item['id']: item['bbox'] for item in raw_objects
        if item['type'] != 'table'
    }
    assert validate_object_distances(simulation_objects) == []
    assert all(
        item['bbox'][2] <= perception['image_size'][0]
        and item['bbox'][3] <= perception['image_size'][1]
        for item in simulation_objects
    )
    assert raw_boxes
    assert simulation_boxes


def test_image_2_simulation_stays_separated_after_apple_placement():
    _, environment, _ = perceive_image(
        IMAGE_2,
        vision_backend='opencv',
    )
    simulation_objects = environment['simulation_objects']
    allowed_containment = environment['simulation_placement'][
        'containment'
    ]
    instances = [
        ObjectInstance(
            instance_id=item['id'],
            object_type=item['type'],
            color=item['color'],
            position=tuple(item['position']),
        )
        for item in simulation_objects
        if item['type'] != 'table'
    ]
    perceived_scene = SimpleNamespace(
        source='perception',
        scenario_id='image_2_test',
        objects=instances,
        allowed_containment=allowed_containment,
        simulation_placement_applied=True,
    )

    import pybullet as pybullet

    robot_id, object_table = create_scene(
        spec=perceived_scene,
        connection_mode=pybullet.DIRECT,
    )
    try:
        cup = object_table['cup']
        basket = object_table['basket']
        apple = object_table['red_apple_1']
        expected_positions = {
            item['id']: item['position']
            for item in simulation_objects
        }
        for object_id, object_spec in object_table.items():
            assert object_spec.position == expected_positions[object_id]
        cup_start = list(
            pybullet.getBasePositionAndOrientation(cup.body_id)[0]
        )
        basket_xy = basket.position[:2]

        conflicts = validate_object_distances([
            {
                'id': object_id,
                'type': object_spec.object_type,
                'position': object_spec.position,
            }
            for object_id, object_spec in object_table.items()
        ])
        assert conflicts == []
        assert math.dist(cup_start[:2], basket_xy) > 0.20

        state = {'held_object': None, 'constraint_id': None}
        handlers = build_skill_handlers(robot_id, object_table, state)
        actions = [
            SkillAction('move_to', {'object': 'red_apple_1'}),
            SkillAction('pick', {'object': 'red_apple_1'}),
            SkillAction('move_to', {'object': 'basket'}),
            SkillAction(
                'place',
                {'object': 'red_apple_1', 'target': 'basket'},
            ),
        ]
        for action in actions:
            handlers[action.skill](action, None)

        cup_end = pybullet.getBasePositionAndOrientation(cup.body_id)[0]
        apple_end = pybullet.getBasePositionAndOrientation(
            apple.body_id
        )[0]
        assert math.dist(cup_start[:2], cup_end[:2]) < 0.01
        assert math.dist(cup_end[:2], basket_xy) > 0.20
        assert math.dist(apple_end[:2], basket_xy) < 0.01
        assert state['held_object'] is None
    finally:
        pybullet.disconnect()
