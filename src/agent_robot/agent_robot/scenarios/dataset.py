"""Render reproducible RGB scenes and ground-truth perception labels."""

import argparse
import json
import os

import pybullet as p
from PIL import Image

from agent_robot.pybullet_robot import create_scene
from agent_robot.scenarios.generator import generate_scenario
from agent_robot.scenarios.spec import SCENARIO_CATEGORIES

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
CAMERA_EYE = (0.60, 0.0, 1.65)
CAMERA_TARGET = (0.60, 0.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)
CAMERA_FOV = 50.0
CAMERA_NEAR = 0.05
CAMERA_FAR = 3.0


def _object_boxes(segmentation, width, height, body_to_object):
    boxes = {}
    for y in range(height):
        for x in range(width):
            encoded = int(segmentation[y, x])
            if encoded < 0:
                continue
            body_id = encoded & 0xFFFFFF
            object_id = body_to_object.get(body_id)
            if object_id is None:
                continue
            bounds = boxes.get(object_id)
            if bounds is None:
                boxes[object_id] = [x, y, x, y]
            else:
                bounds[0] = min(bounds[0], x)
                bounds[1] = min(bounds[1], y)
                bounds[2] = max(bounds[2], x)
                bounds[3] = max(bounds[3], y)

    return {
        object_id: [
            bounds[0],
            bounds[1],
            bounds[2] - bounds[0] + 1,
            bounds[3] - bounds[1] + 1,
        ]
        for object_id, bounds in boxes.items()
    }


def render_scene(spec, image_path, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Render a SceneSpec and save image-space plus metric ground truth."""
    if p.isConnected():
        raise RuntimeError(
            'Disconnect existing PyBullet clients before rendering a scene.'
        )
    if width <= 0 or height <= 0:
        raise ValueError('Image width and height must be positive.')

    robot_id, object_table = create_scene(
        spec=spec,
        connection_mode=p.DIRECT,
    )
    try:
        for link_id in range(-1, p.getNumJoints(robot_id)):
            p.changeVisualShape(
                robot_id,
                link_id,
                rgbaColor=[0.0, 0.0, 0.0, 0.0],
            )

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=CAMERA_EYE,
            cameraTargetPosition=CAMERA_TARGET,
            cameraUpVector=CAMERA_UP,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV,
            aspect=float(width) / height,
            nearVal=CAMERA_NEAR,
            farVal=CAMERA_FAR,
        )
        image = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )
        rgba, segmentation = image[2], image[4]
        output_image = Image.frombytes('RGBA', (width, height), bytes(rgba))
        output_image.convert('RGB').save(image_path)

        body_to_object = {}
        for object_id, object_spec in object_table.items():
            body_ids = object_spec.body_ids or []
            if object_spec.body_id >= 0:
                body_ids = body_ids + [object_spec.body_id]
            for body_id in body_ids:
                body_to_object[body_id] = object_id

        boxes = _object_boxes(
            segmentation,
            width,
            height,
            body_to_object,
        )
        labels = []
        for instance in spec.objects:
            object_spec = object_table[instance.instance_id]
            position = list(instance.position)
            if object_spec.body_id >= 0:
                position = list(
                    p.getBasePositionAndOrientation(
                        object_spec.body_id
                    )[0]
                )
            labels.append({
                'id': instance.instance_id,
                'name': instance.object_type,
                'type': instance.object_type,
                'color': instance.color,
                'position': position,
                'graspable': instance.graspable,
                'receptacle': instance.receptacle,
                'visible': instance.visible,
                'bbox_xywh': boxes.get(instance.instance_id),
            })

        return {
            'schema_version': '1.0',
            'scenario_id': spec.scenario_id,
            'seed': spec.seed,
            'command': spec.command,
            'image': os.path.basename(image_path),
            'image_size': [width, height],
            'camera': {
                'eye': list(CAMERA_EYE),
                'target': list(CAMERA_TARGET),
                'up': list(CAMERA_UP),
                'fov_degrees': CAMERA_FOV,
                'near': CAMERA_NEAR,
                'far': CAMERA_FAR,
                'view_matrix': list(view_matrix),
                'projection_matrix': list(projection_matrix),
            },
            'objects': labels,
            'scene_spec': spec.to_dict(),
        }
    finally:
        p.disconnect()


def generate_dataset(
    output_dir,
    count,
    seed=0,
    category=None,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
):
    """Generate seeded scene directories containing RGB and scene JSON."""
    if count <= 0:
        raise ValueError('Scene count must be positive.')
    if category is not None and category not in SCENARIO_CATEGORIES:
        raise ValueError('Unknown scenario category: {}'.format(category))

    categories = [category] if category else list(SCENARIO_CATEGORIES)
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for index in range(count):
        scene_category = categories[index % len(categories)]
        scene_seed = seed + index
        spec = generate_scenario(
            scene_category,
            seed=scene_seed,
            scenario_id='scene_{:04d}'.format(index + 1),
        )
        scene_dir = os.path.join(
            output_dir,
            'scene_{:04d}'.format(index + 1),
        )
        os.makedirs(scene_dir, exist_ok=True)
        image_name = 'rgb.png'
        image_path = os.path.join(scene_dir, image_name)
        labels = render_scene(spec, image_path, width, height)
        labels_path = os.path.join(scene_dir, 'scene.json')
        with open(labels_path, 'w', encoding='utf-8') as output_file:
            json.dump(labels, output_file, ensure_ascii=False, indent=2)
        outputs.append((image_path, labels_path))
    return outputs


def main(args=None):
    """Generate a local RGB scene dataset."""
    parser = argparse.ArgumentParser(
        description='Render seeded tabletop scenes and labels.'
    )
    parser.add_argument('--output-dir', default='datasets/vision_scenes')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--category',
        choices=SCENARIO_CATEGORIES,
        default=None,
    )
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    options = parser.parse_args(args=args)
    outputs = generate_dataset(
        options.output_dir,
        options.count,
        seed=options.seed,
        category=options.category,
        width=options.width,
        height=options.height,
    )
    print('Generated {} scenes in {}'.format(
        len(outputs), os.path.abspath(options.output_dir)
    ))


if __name__ == '__main__':
    main()
