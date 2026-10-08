import json
import os
from types import SimpleNamespace

import rclpy
from std_msgs.msg import String

from agent_robot.scenarios.benchmark import (
    benchmark_summary,
    generate_benchmark_episode,
    generate_benchmark_episodes,
)
from agent_robot.perception.pipeline import perceive_image
from agent_robot.scene_graph import resolve_instance
from agent_robot.scenarios.spec import validate
from agent_robot.task_planner import TaskPlanner


def test_benchmark_episode_is_seeded_and_valid():
    first = generate_benchmark_episode(17)
    second = generate_benchmark_episode(17)

    assert first.to_dict() == second.to_dict()
    validate(first)
    assert first.goal.params['target_id'] in {
        item.instance_id for item in first.objects
    }
    assert first.goal.params['container_id'] in {
        item.instance_id for item in first.objects
    }
    assert len(first.objects) >= 3
    assert {item.object_type for item in first.objects} <= {
        'apple', 'cup', 'basket', 'bin'
    }


def test_benchmark_dataset_contains_scene_and_task(tmp_path):
    episodes = generate_benchmark_episodes(
        str(tmp_path), count=2, seed=4, width=320, height=240
    )

    assert len(episodes) == 2
    for episode in episodes:
        assert os.path.isfile(episode['image_path'])
        with open(episode['spec_path'], encoding='utf-8') as handle:
            scene = json.load(handle)
        with open(episode['task_path'], encoding='utf-8') as handle:
            task = json.load(handle)
        assert scene['scenario_id'] == task['episode_id']
        assert task['instruction']
        assert task['goal']['target_id'] in {
            obj['id'] for obj in scene['objects']
        }
        _, _, graph = perceive_image(
            episode['image_path'], vision_backend='synthetic_opencv'
        )
        assert task['goal']['target_id'] in graph.objects
        assert task['goal']['container_id'] in graph.objects
        if task['goal']['container_id'].endswith('_bin_1'):
            assert resolve_instance('box', graph) == (
                task['goal']['container_id']
            )


def test_benchmark_summary_reports_success_and_failure_metrics():
    summary = benchmark_summary([
        {
            'success': True,
            'failure_type': None,
            'failures': [{
                'failure_code': 'grasp_failed',
            }],
            'recovered': True,
            'trajectory_length': 4,
            'execution_time_seconds': 1.2,
        },
        {
            'success': False,
            'failure_type': 'planning_failure',
            'failures': [],
            'recovered': False,
            'trajectory_length': 0,
            'execution_time_seconds': 0.4,
        },
    ])

    assert summary['episodes'] == 2
    assert summary['success_rate'] == 0.5
    assert summary['failure_counts']['planning_failure'] == 1
    assert summary['mean_trajectory_length'] == 2.0
    assert summary['mean_execution_time_seconds'] == 0.8
    assert summary['task_success_rate'] == 0.5
    assert summary['failure_recovery_rate'] == 1.0


def test_planner_only_replans_grasp_failure_once():
    class FakePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(json.loads(message.data))

    class FakeLogger:
        def warning(self, message):
            pass

        def info(self, message):
            pass

        def error(self, message):
            pass

    publisher = FakePublisher()
    node = SimpleNamespace(
        replan_counts={},
        planner_mode='mock',
        get_logger=lambda: FakeLogger(),
        plan_publisher=publisher,
        generate_plan=lambda command, failure_context: {
            'feasible': True,
            'actions': [],
            'replans': failure_context['replans'],
        },
    )

    failure = String()
    failure.data = json.dumps({
        'status': 'failed',
        'command': 'put apple into basket',
        'failure_code': 'grasp_failed',
        'action': {'skill': 'pick', 'object': 'apple'},
    })
    TaskPlanner.status_callback(node, failure)
    TaskPlanner.status_callback(node, failure)

    unrelated = String()
    unrelated.data = json.dumps({
        'status': 'failed',
        'command': 'put apple into basket',
        'failure_code': 'execution_failed',
        'action': {'skill': 'place', 'object': 'apple'},
    })
    TaskPlanner.status_callback(node, unrelated)

    assert len(publisher.messages) == 1
    assert publisher.messages[0]['status'] == 'replanned'
    assert publisher.messages[0]['replans'] == 1


def test_planner_waits_for_perception_before_rejecting_command():
    rclpy.init(args=[])
    planner = TaskPlanner()
    published = []
    planner.planner_mode = 'mock'
    planner._publish_plan = published.append
    command = String()
    command.data = 'put apple into basket'

    try:
        planner.command_callback(command)
        assert planner.pending_command == command.data
        assert published == []

        _, environment, _ = perceive_image(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'images',
                '2.png',
            ),
            vision_backend='opencv',
        )
        message = String()
        message.data = json.dumps(environment)
        planner.environment_callback(message)

        assert planner.pending_command is None
        assert len(published) == 1
        assert published[0]['feasible'] is True
        assert published[0]['planner'] == 'mock'
    finally:
        planner.destroy_node()
        rclpy.shutdown()
