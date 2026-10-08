"""Generate rendered, task-labeled episodes for the M5 vision benchmark."""

import json
import os
import random

from agent_robot.scenarios.dataset import render_scene
from agent_robot.scenarios.generator import (
    _LayoutBuilder,
    _build_spec,
    _instance,
)
from agent_robot.scenarios.spec import (
    CATEGORY_TARGET_PICK,
)
from agent_robot.scene_graph import SCENE_COLORS, build_instance_id

APPLE_COLORS = ('red', 'green')
COLOR_WORDS = {'red': 'red', 'green': 'green'}


def generate_benchmark_episode(seed, episode_id=None):
    """Create a reproducible scene and grounded pick-and-place instruction."""
    rng = random.Random(seed)
    episode_id = episode_id or 'episode_{:04d}'.format(seed + 1)
    layout = _LayoutBuilder(rng)
    target_type = rng.choice(('apple', 'cup'))
    target_color = rng.choice(APPLE_COLORS) if target_type == 'apple' else None
    use_box = rng.random() < 0.2
    container_type = 'bin' if use_box else 'basket'

    objects = []
    distractor_count = rng.randint(0, 2)
    container_position = layout.place_container(container_type)
    add_extra_container = rng.random() < 0.35
    extra_type = 'basket' if container_type == 'bin' else 'bin'
    extra_position = (
        layout.place_container(extra_type)
        if add_extra_container else None
    )

    if target_type == 'apple':
        target_id = build_instance_id('apple', target_color, 1)
        objects.append(_instance(
            'apple',
            layout.sample_object('apple'),
            instance_id=target_id,
            color=target_color,
        ))
        other_color = 'green' if target_color == 'red' else 'red'
        for index in range(distractor_count):
            objects.append(_instance(
                'apple',
                layout.sample_object('apple'),
                instance_id=build_instance_id('apple', other_color, index + 1),
                color=other_color,
            ))
        if rng.random() < 0.65:
            objects.append(_instance('cup', layout.sample_object('cup')))
    else:
        target_id = 'cup'
        objects.append(_instance('cup', layout.sample_object('cup')))
        apple_color = rng.choice(APPLE_COLORS)
        for index in range(max(1, distractor_count)):
            objects.append(_instance(
                'apple',
                layout.sample_object('apple'),
                instance_id=build_instance_id('apple', apple_color, index + 1),
                color=apple_color,
            ))

    if container_type == 'bin':
        container_color = rng.choice(SCENE_COLORS)
        container_id = build_instance_id('bin', container_color, 1)
        objects.append(_instance(
            'bin', container_position, instance_id=container_id,
            color=container_color,
        ))
    else:
        container_id = 'basket'
        objects.append(_instance('basket', container_position))

    # Add a non-target receptacle as a distractor in some episodes.
    if add_extra_container:
        if extra_type == 'basket':
            objects.append(_instance('basket', extra_position))
        else:
            extra_color = next(
                color for color in SCENE_COLORS
                if build_instance_id('bin', color, 1) != container_id
            )
            objects.append(_instance(
                'bin',
                extra_position,
                instance_id=build_instance_id('bin', extra_color, 1),
                color=extra_color,
            ))

    target_name = target_type
    if target_type == 'apple':
        target_name = '{} apple'.format(COLOR_WORDS[target_color])
    container_name = 'box' if use_box else 'basket'
    command = 'put {} into {}'.format(target_name, container_name)
    spec = _build_spec(
        CATEGORY_TARGET_PICK,
        seed,
        command,
        objects,
        {'target_id': target_id, 'container_id': container_id},
        scenario_id=episode_id,
    )
    return spec


def generate_benchmark_episodes(
    output_dir, count, seed=0, width=640, height=480
):
    """Write each initial spec, rendered RGB image, and task label."""
    if count <= 0:
        raise ValueError('Episode count must be positive.')
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    episodes = []

    for index in range(count):
        episode_id = 'episode_{:04d}'.format(index + 1)
        episode_dir = os.path.join(output_dir, episode_id)
        os.makedirs(episode_dir, exist_ok=True)
        spec = generate_benchmark_episode(seed + index, episode_id)
        image_path = os.path.join(episode_dir, 'rgb.png')
        render_scene(spec, image_path, width, height)
        spec_path = os.path.join(episode_dir, 'scene_spec.json')
        with open(spec_path, 'w', encoding='utf-8') as handle:
            json.dump(spec.to_dict(), handle, ensure_ascii=False, indent=2)
        task_path = os.path.join(episode_dir, 'task.json')
        with open(task_path, 'w', encoding='utf-8') as handle:
            json.dump({
                'schema_version': '1.0',
                'episode_id': episode_id,
                'instruction': spec.command,
                'goal': {
                    'target_id': spec.goal.params['target_id'],
                    'container_id': spec.goal.params['container_id'],
                },
                'scene_spec': os.path.basename(spec_path),
                'image': os.path.basename(image_path),
            }, handle, ensure_ascii=False, indent=2)
        episodes.append({
            'episode_id': episode_id,
            'directory': episode_dir,
            'image_path': image_path,
            'spec_path': spec_path,
            'task_path': task_path,
        })
    return episodes


def benchmark_summary(rows):
    """Summarize episode outcomes with explicit failure categories."""
    total = len(rows)
    succeeded = sum(1 for row in rows if row.get('success') is True)
    failures = {
        category: sum(
            1 for row in rows if row.get('failure_type') == category
        )
        for category in (
            'perception_failure',
            'planning_failure',
            'invalid_action',
            'execution_failure',
        )
    }
    recovery_episodes = [
        row for row in rows
        if any(
            failure.get('failure_code') == 'grasp_failed'
            for failure in row.get('failures', [])
        )
    ]
    rates = {
        'task_success_rate': succeeded / float(total) if total else 0.0,
        'perception_failure_rate': (
            failures['perception_failure'] / float(total) if total else 0.0
        ),
        'planning_failure_rate': (
            failures['planning_failure'] / float(total) if total else 0.0
        ),
        'invalid_action_rate': (
            failures['invalid_action'] / float(total) if total else 0.0
        ),
        'execution_failure_rate': (
            failures['execution_failure'] / float(total) if total else 0.0
        ),
        'failure_recovery_rate': (
            sum(1 for row in recovery_episodes if row.get('recovered'))
            / float(len(recovery_episodes))
            if recovery_episodes else None
        ),
    }
    durations = [
        row['execution_time_seconds']
        for row in rows
        if isinstance(row.get('execution_time_seconds'), (int, float))
    ]
    trajectory_lengths = [
        row['trajectory_length']
        for row in rows
        if isinstance(row.get('trajectory_length'), (int, float))
    ]
    return {
        'episodes': total,
        'successes': succeeded,
        'success_rate': round(rates['task_success_rate'], 4),
        **{
            key: round(value, 4) if value is not None else None
            for key, value in rates.items()
        },
        'failure_counts': failures,
        'mean_trajectory_length': (
            round(sum(trajectory_lengths) / float(len(trajectory_lengths)), 4)
            if trajectory_lengths else 0.0
        ),
        'mean_execution_time_seconds': (
            round(sum(durations) / float(len(durations)), 4)
            if durations else 0.0
        ),
        'average_trajectory_length': (
            round(sum(trajectory_lengths) / float(len(trajectory_lengths)), 4)
            if trajectory_lengths else 0.0
        ),
        'average_execution_time_seconds': (
            round(sum(durations) / float(len(durations)), 4)
            if durations else 0.0
        ),
    }


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate rendered scenes and tasks for M5.'
    )
    parser.add_argument('--output-dir', default='datasets/m5_benchmark')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    options = parser.parse_args(args=args)
    episodes = generate_benchmark_episodes(
        options.output_dir, options.count, seed=options.seed
    )
    print('Generated {} episodes in {}'.format(
        len(episodes), os.path.abspath(options.output_dir)
    ))
