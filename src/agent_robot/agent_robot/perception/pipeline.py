"""Pure image-to-structured-scene pipeline shared by ROS and batch runs."""

import os

import cv2

from agent_robot.perception.coordinate_mapper import (
    CoordinateMapper,
    clamp_bbox,
    correct_simulation_placements,
)
from agent_robot.perception.object_detector import detect_image
from agent_robot.perception.object_detector import detect_generated_scene
from agent_robot.perception.vision_api import detect_image_with_deepseek
from agent_robot.scene_graph import SceneGraph


def perceive_image(image_path, mapper=None, vision_backend='opencv'):
    """Detect, map, and build a scene graph without reading label files."""
    image_path = os.path.abspath(os.path.expanduser(image_path))
    if vision_backend == 'opencv':
        detections = detect_image(image_path)
        perception_source = 'opencv_perception'
    elif vision_backend == 'synthetic_opencv':
        detections = detect_generated_scene(image_path)
        perception_source = 'synthetic_opencv_perception'
    elif vision_backend == 'deepseek':
        detections = detect_image_with_deepseek(image_path)
        perception_source = 'deepseek_vision'
    else:
        raise ValueError(
            'Unsupported vision_backend: {}'.format(vision_backend)
        )
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not read image: {}'.format(image_path))
    image_height, image_width = image.shape[:2]
    mapper = mapper or CoordinateMapper()
    scenario_id = os.path.splitext(os.path.basename(image_path))[0]

    environment_objects = []
    detection_warnings = []
    for detection in detections:
        if detection['type'] == 'table':
            continue
        try:
            position = mapper.map_bbox(
                detection['bbox'],
                (image_width, image_height),
                detection['type'],
            )
        except ValueError as error:
            detection_warnings.append(
                'Skipped {} detection with invalid image bbox {}: {}'
                .format(
                    detection.get('id', detection.get('type', 'unknown')),
                    detection.get('bbox'),
                    error,
                )
            )
            continue
        environment_objects.append({
            'id': detection['id'],
            'name': detection['name'],
            'type': detection['type'],
            'color': detection['color'],
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'position': position,
            'support': 'table',
            'visible': True,
        })

    environment = {
        'source': 'perception',
        'scenario_id': scenario_id,
        'image_path': image_path,
        'world_revision': 0,
        'objects': environment_objects,
    }
    simulation_inputs = []
    for item in environment_objects:
        simulation_item = dict(item)
        simulation_item['bbox'] = clamp_bbox(
            item['bbox'],
            (image_width, image_height),
        )
        simulation_inputs.append(simulation_item)
    placement = correct_simulation_placements(simulation_inputs)
    environment['simulation_objects'] = placement['objects']
    environment['simulation_placement'] = {
        'warnings': detection_warnings + placement['warnings'],
        'adjustments': placement['adjustments'],
        'containment': placement['containment'],
    }
    return {
        'schema_version': '1.0',
        'source': perception_source,
        'scenario_id': scenario_id,
        'image_path': image_path,
        'image_size': [image_width, image_height],
        'objects': detections,
    }, environment, SceneGraph.build_from_perception(environment)
