"""DeepSeek multimodal image detector using its OpenAI-compatible API."""

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

SUPPORTED_TYPES = {'apple', 'basket', 'cup', 'table'}
VISION_PROMPT = """Inspect the attached tabletop image.
Return only a JSON object with this exact shape:
{"objects":[{"type":"apple","color":"red","bbox":[x1,y1,x2,y2],
 "confidence":0.0}]}

Detect only clearly visible apples, baskets, cups and the table. Use
pixel coordinates relative to the original image, with x1,y1 inclusive
and x2,y2 exclusive. Do not invent hidden objects. Apples should include
their visible color. Basket and cup color may be null. Confidence must
be between 0 and 1. The objects value must be an array; return an empty
array when no supported object is visible."""


def detect_image_with_deepseek(
    image_path,
    api_key=None,
    api_url=None,
    model=None,
):
    """Return validated pixel detections from a single image."""
    key = api_key or os.environ.get('DEEPSEEK_VISION_API_KEY')
    key = key or os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise ValueError(
            'Set DEEPSEEK_API_KEY or DEEPSEEK_VISION_API_KEY for '
            'DeepSeek vision.'
        )

    endpoint = api_url or os.environ.get(
        'DEEPSEEK_VISION_API_URL',
        'https://api.deepseek.com/chat/completions',
    )
    model_name = model or os.environ.get(
        'DEEPSEEK_VISION_MODEL',
        'deepseek-flash',
    )
    image_path = os.path.abspath(os.path.expanduser(image_path))
    mime_type = mimetypes.guess_type(image_path)[0]
    if mime_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise ValueError(
            'Vision API supports PNG, JPEG or WEBP images; got {}.'.format(
                mime_type or 'unknown image type'
            )
        )
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('ascii')

    request_data = {
        'model': model_name,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': VISION_PROMPT},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': 'data:{};base64,{}'.format(
                                mime_type, encoded_image
                            ),
                        },
                    },
                ],
            },
        ],
        'response_format': {'type': 'json_object'},
        'stream': False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_data).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(key),
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        raise RuntimeError(
            'DeepSeek vision API returned HTTP {}: {}'.format(
                error.code, detail
            )
        )
    except urllib.error.URLError as error:
        raise RuntimeError(
            'Could not reach the DeepSeek vision API: {}'.format(error)
        )

    try:
        content = payload['choices'][0]['message']['content']
        detections = json.loads(content)['objects']
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            'DeepSeek vision response did not contain valid '
            '{"objects": [...]} JSON.'
        ) from error

    if not isinstance(detections, list):
        raise ValueError('DeepSeek vision "objects" must be an array.')
    return _validate_detections(detections)


def _validate_detections(detections):
    validated = []
    instance_counts = {}
    for index, detection in enumerate(detections, start=1):
        if not isinstance(detection, dict):
            raise ValueError(
                'Vision detection {} must be a JSON object.'.format(index)
            )
        object_type = detection.get('type')
        if object_type not in SUPPORTED_TYPES:
            continue
        bbox = detection.get('bbox')
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in bbox
            )
        ):
            raise ValueError(
                'Vision detection {} has an invalid bbox.'.format(index)
            )
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                'Vision detection {} has a non-positive bbox.'.format(
                    index
                )
            )
        confidence = detection.get('confidence', 0.0)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(
                'Vision detection {} has invalid confidence.'.format(index)
            )
        color = detection.get('color')
        if color is not None and not isinstance(color, str):
            raise ValueError(
                'Vision detection {} has an invalid color.'.format(index)
            )
        name = object_type
        if object_type == 'apple' and color:
            name = '{}_apple'.format(color.lower())
        if object_type == 'apple' and color and color.lower() in {
            'red', 'blue', 'green', 'yellow'
        }:
            count_key = (object_type, color.lower())
            instance_counts[count_key] = instance_counts.get(count_key, 0) + 1
            object_id = '{}_{}_{}'.format(
                color.lower(), object_type, instance_counts[count_key]
            )
        else:
            object_id = object_type
        validated.append({
            'id': object_id,
            'name': name,
            'type': object_type,
            'color': color,
            'bbox': [x1, y1, x2, y2],
            'confidence': float(confidence),
        })
    return validated
