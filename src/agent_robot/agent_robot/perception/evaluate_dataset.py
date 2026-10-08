"""Run perception over an image directory and export scene graph JSON."""

import argparse
import glob
import json
import os
import time

from agent_robot.perception.pipeline import perceive_image

IMAGE_PATTERNS = ('*.png', '*.jpg', '*.jpeg', '*.webp')


def evaluate_directory(image_dir, output_dir, mapper=None):
    image_paths = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_paths = [
        path for path in image_paths
        if os.path.basename(path) != 'scene.png'
    ]
    image_paths.sort()
    if not image_paths:
        raise ValueError(
            'No supported images found in {}'.format(image_dir)
        )

    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'schema_version': '1.0',
        'image_count': len(image_paths),
        'mapping_note': (
            'Positions are affine estimates; accuracy requires camera/table '
            'calibration and annotated image ground truth.'
        ),
        'images': [],
    }
    for image_path in image_paths:
        started = time.monotonic()
        perception, _, graph = perceive_image(image_path, mapper)
        elapsed = time.monotonic() - started
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        result_path = os.path.join(
            output_dir,
            '{}_scene_graph.json'.format(base_name),
        )
        with open(result_path, 'w', encoding='utf-8') as output_file:
            json.dump(
                graph.to_dict(),
                output_file,
                ensure_ascii=False,
                indent=2,
            )
        counts = {}
        for detection in perception['objects']:
            object_type = detection['type']
            counts[object_type] = counts.get(object_type, 0) + 1
        summary['images'].append({
            'image': image_name,
            'scene_graph': os.path.basename(result_path),
            'detected_objects': counts,
            'elapsed_seconds': elapsed,
        })

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as output_file:
        json.dump(
            summary,
            output_file,
            ensure_ascii=False,
            indent=2,
        )
    return summary


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate OpenCV perception on a directory of images.'
    )
    parser.add_argument(
        '--image-dir',
        default='src/agent_robot/images',
    )
    parser.add_argument('--output-dir', default='results')
    options = parser.parse_args(args=args)
    summary = evaluate_directory(options.image_dir, options.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
