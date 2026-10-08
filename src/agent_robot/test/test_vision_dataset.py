"""Tests for reproducible image and label dataset generation."""

import json
import os
import sys

import pybullet as p
import pytest
from PIL import Image
import rclpy

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scenarios.dataset import generate_dataset  # noqa: E402
from agent_robot.perception.coordinate_mapper import (  # noqa: E402
    CoordinateMapper,
)
from agent_robot.perception.evaluate_dataset import (  # noqa: E402
    evaluate_directory,
)
from agent_robot.perception.object_detector import detect_image  # noqa: E402
from agent_robot.perception.pipeline import perceive_image  # noqa: E402
from agent_robot.scene_graph import resolve_instance  # noqa: E402
from agent_robot.task_planner import TaskPlanner  # noqa: E402

IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'images',
)


def test_dataset_contains_rendered_image_and_ground_truth(tmp_path):
    outputs = generate_dataset(
        str(tmp_path),
        count=1,
        seed=41,
        category='target_pick',
        width=320,
        height=240,
    )

    assert len(outputs) == 1
    image_path, labels_path = outputs[0]
    assert not p.isConnected()
    with Image.open(image_path) as image:
        assert image.size == (320, 240)
        assert image.mode == 'RGB'

    with open(labels_path, encoding='utf-8') as labels_file:
        labels = json.load(labels_file)
    assert labels['scenario_id'] == 'scene_0001'
    assert labels['seed'] == 41
    assert labels['image'] == 'rgb.png'
    assert labels['image_size'] == [320, 240]
    assert labels['camera']['view_matrix']
    assert labels['camera']['projection_matrix']
    assert labels['objects']
    for obj in labels['objects']:
        assert len(obj['position']) == 3
        assert obj['bbox_xywh'] is not None
        x, y, width, height = obj['bbox_xywh']
        assert 0 <= x < 320
        assert 0 <= y < 240
        assert width > 0
        assert height > 0


def test_dataset_rejects_invalid_arguments(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        generate_dataset(str(tmp_path), count=0)
    with pytest.raises(ValueError, match='Unknown scenario category'):
        generate_dataset(str(tmp_path), count=1, category='unknown')


def test_detector_finds_primary_objects_in_all_six_photos():
    for image_index in range(1, 7):
        detections = detect_image(
            os.path.join(IMAGE_DIR, '{}.png'.format(image_index))
        )
        detected_types = {item['type'] for item in detections}
        assert {'apple', 'basket', 'table'} <= detected_types
        for item in detections:
            x1, y1, x2, y2 = item['bbox']
            assert x2 > x1
            assert y2 > y1
            assert 0.0 <= item['confidence'] <= 1.0


def test_image_pipeline_builds_graph_with_bbox_and_world_position():
    image_path = os.path.join(IMAGE_DIR, '4.png')
    perception, environment, graph = perceive_image(
        image_path,
        CoordinateMapper(
            world_x=(0.4, 0.7),
            world_y=(-0.2, 0.2),
        ),
    )
    assert perception['source'] == 'opencv_perception'
    assert environment['source'] == 'perception'
    apple_nodes = [
        node for node in graph.to_dict()['objects']
        if node['name'].endswith('_apple_1')
        or node['name'].endswith('_apple_2')
    ]
    assert len(apple_nodes) == 2
    assert resolve_instance('red apple', graph) == 'red_apple_1'
    assert resolve_instance('green apple', graph) == 'green_apple_2'
    for node in apple_nodes:
        assert node['bbox'] is not None
        assert node['confidence'] is not None
        assert len(node['position']) == 3
        assert graph.supports[node['name']] == ('on', 'table')


def test_fixed_camera_coordinate_mapper_maps_image_edges():
    mapper = CoordinateMapper(
        world_x=(0.3, 0.8),
        world_y=(-0.4, 0.4),
    )
    assert mapper.pixel_to_world((0, 0), (100, 100), 'apple') == [
        0.3, 0.4, 0.06
    ]
    assert mapper.pixel_to_world((100, 100), (100, 100), 'basket') == [
        0.8, -0.4, 0.05
    ]


def test_evaluation_exports_one_graph_per_real_image(tmp_path):
    summary = evaluate_directory(IMAGE_DIR, str(tmp_path))
    assert summary['image_count'] == 6
    for image_index in range(1, 7):
        path = tmp_path / '{}_scene_graph.json'.format(image_index)
        with open(path, encoding='utf-8') as graph_file:
            graph = json.load(graph_file)
        assert graph['source'] == 'perception'
        assert graph['objects']
    assert (tmp_path / 'summary.json').is_file()


def test_scene_grounded_planner_generates_or_rejects_without_manual_plan():
    rclpy.init(args=[])
    planner = TaskPlanner()
    try:
        perception, environment, graph = perceive_image(
            os.path.join(IMAGE_DIR, '2.png')
        )
        planner.current_environment = environment
        planner.latest_scene_graph = graph
        planner.planner_mode = 'mock'
        plan = planner.generate_plan('Put the apple into the basket')
        assert plan['feasible'] is True
        assert plan['planner'] == 'mock'
        assert plan['image_path'] == environment['image_path']
        assert plan['actions'][0]['object'] == 'red_apple_1'
        assert plan['actions'][-1]['target'] == 'basket'

        missing_object = planner.generate_plan(
            'Put the banana into the basket'
        )
        assert missing_object['feasible'] is False
        assert 'banana' in missing_object['failure_reason']
        assert 'banana' in missing_object['reason']
        assert missing_object['actions'] == []

        environment['objects'] = [
            obj for obj in environment['objects']
            if obj['type'] != 'basket'
        ]
        planner.current_environment = environment
        planner.latest_scene_graph = None
        rejected = planner.generate_plan(
            'Put the apple into the basket'
        )
        assert rejected['feasible'] is False
        assert 'basket' in rejected['reason']
        assert rejected['actions'] == []
    finally:
        planner.destroy_node()
        rclpy.shutdown()
