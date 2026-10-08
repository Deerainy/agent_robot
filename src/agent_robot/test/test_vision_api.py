import os
import sys
import json
import tempfile
from types import SimpleNamespace
from unittest import mock

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.perception.vision_api import (  # noqa: E402
    _validate_detections,
    detect_image_with_deepseek,
)
from agent_robot.scenarios.spec import ObjectInstance  # noqa: E402
from agent_robot.scene_graph import SceneGraph  # noqa: E402


def test_validate_vision_objects_and_names():
    detections = _validate_detections([
        {
            'type': 'apple',
            'color': 'red',
            'bbox': [10, 20, 40, 55],
            'confidence': 0.9,
        },
        {
            'type': 'basket',
            'color': None,
            'bbox': [60.2, 15, 110, 90],
            'confidence': 0.8,
        },
        {
            'type': 'person',
            'bbox': [0, 0, 50, 50],
            'confidence': 0.99,
        },
    ])

    assert len(detections) == 2
    assert detections[0]['id'] == 'red_apple_1'
    assert detections[0]['bbox'] == [10, 20, 40, 55]
    assert detections[1]['id'] == 'basket'


def test_vision_instance_ids_match_scene_graph_and_simulator():
    detections = _validate_detections([
        {
            'type': 'apple',
            'color': 'red',
            'bbox': [10, 20, 40, 55],
            'confidence': 0.9,
        },
        {
            'type': 'basket',
            'bbox': [60, 15, 110, 90],
            'confidence': 0.8,
        },
    ])
    environment = {
        'source': 'perception',
        'scenario_id': 'test',
        'world_revision': 0,
        'objects': [
            dict(
                item,
                position=(
                    [0.45, -0.20, 0.06]
                    if item['type'] == 'apple'
                    else [0.72, 0.20, 0.05]
                ),
            )
            for item in detections
        ],
    }
    graph = SceneGraph.build_from_perception(environment)
    instances = tuple(
        ObjectInstance(
            instance_id=item['id'],
            object_type=item['type'],
            color=item['color'],
            position=(
                (0.45, -0.20, 0.06)
                if item['type'] == 'apple'
                else (0.72, 0.20, 0.05)
            ),
        )
        for item in detections
    )

    assert set(graph.objects) == {'red_apple_1', 'basket'}
    assert {instance.instance_id for instance in instances} == \
        set(graph.objects)

    import pybullet as pybullet
    from agent_robot.pybullet_robot import create_scene

    scene = SimpleNamespace(scenario_id='vision_test', objects=instances)
    _, object_table = create_scene(
        spec=scene,
        connection_mode=pybullet.DIRECT,
    )
    try:
        assert set(object_table) == set(graph.objects)
    finally:
        pybullet.disconnect()


def test_validate_vision_detections_rejects_invalid_bbox():
    try:
        _validate_detections([{
            'type': 'cup',
            'bbox': [4, 8, 2, 9],
            'confidence': 0.5,
        }])
    except ValueError as error:
        assert 'non-positive bbox' in str(error)
    else:
        raise AssertionError('Invalid box was accepted.')


def test_deepseek_request_contains_image_and_uses_shared_key():
    response_body = json.dumps({
        'choices': [{
            'message': {
                'content': json.dumps({
                    'objects': [{
                        'type': 'cup',
                        'bbox': [1, 2, 20, 30],
                        'confidence': 0.8,
                    }]
                })
            }
        }]
    }).encode('utf-8')
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = response_body
    with tempfile.NamedTemporaryFile(suffix='.png') as image_file:
        image_file.write(b'fake png bytes')
        image_file.flush()
        with mock.patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'test-key',
            'DEEPSEEK_VISION_API_KEY': '',
        }):
            with mock.patch(
                'agent_robot.perception.vision_api.urllib.request.urlopen',
                return_value=response,
            ) as urlopen:
                detections = detect_image_with_deepseek(image_file.name)

    request = urlopen.call_args[0][0]
    request_payload = json.loads(request.data.decode('utf-8'))
    assert request_payload['model'] == 'deepseek-flash'
    assert request_payload['messages'][0]['content'][1][
        'image_url'
    ]['url'].startswith('data:image/png;base64,')
    assert request.get_header('Authorization') == 'Bearer test-key'
    assert detections[0]['type'] == 'cup'
