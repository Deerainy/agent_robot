"""Run generated M5 image episodes through the existing ROS vision stack."""

import argparse
import json
import os
import secrets
import signal
import subprocess
import time

import rclpy
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from agent_robot.ros_qos import ENVIRONMENT_STATE_QOS
from agent_robot.scenarios.benchmark import benchmark_summary

SCENE_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class EpisodeIO:
    """Collect the live ROS outputs for one image-grounded episode."""

    def __init__(self, node):
        self.environment = None
        self.scene_graph = None
        self.plan = None
        self.plans = []
        self.statuses = []
        self.trajectory = []
        self.command_publisher = node.create_publisher(
            String, '/user_command', 10
        )
        self.failure_publisher = node.create_publisher(
            String, '/simulate_failure', 10
        )
        self._subscriptions = [
            node.create_subscription(
                String,
                '/environment_state',
                self._environment,
                ENVIRONMENT_STATE_QOS,
            ),
            node.create_subscription(
                String, '/scene_graph', self._scene_graph, SCENE_QOS
            ),
            node.create_subscription(String, '/task_plan', self._plan, 10),
            node.create_subscription(
                String, '/task_status', self._status, 10
            ),
            node.create_subscription(
                String, '/trajectory_point', self._trajectory, 100
            ),
        ]

    def publish_command(self, command):
        message = String()
        message.data = command
        self.command_publisher.publish(message)

    def inject_grasp_failure(self, object_id):
        message = String()
        message.data = json.dumps({
            'failure_code': 'grasp_failed',
            'object': object_id,
        })
        self.failure_publisher.publish(message)

    def close(self, node):
        for subscription in self._subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_publisher(self.command_publisher)
        node.destroy_publisher(self.failure_publisher)

    def _environment(self, message):
        payload = json.loads(message.data)
        if payload.get('source') == 'perception':
            self.environment = payload

    def _scene_graph(self, message):
        self.scene_graph = json.loads(message.data)

    def _plan(self, message):
        self.plan = json.loads(message.data)
        self.plans.append(self.plan)

    def _status(self, message):
        self.statuses.append(json.loads(message.data))

    def _trajectory(self, message):
        self.trajectory.append(json.loads(message.data))


def _spin_until(node, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
    return predicate()


def _terminal_status(statuses):
    for status in reversed(statuses):
        if status.get('status') in ('succeeded', 'rejected'):
            return status
    return None


def _episode_finished(io, failed_at, timeout):
    status = _terminal_status(io.statuses)
    if status is not None:
        return True
    failures = [
        item for item in io.statuses if item.get('status') == 'failed'
    ]
    if not failures:
        return False
    latest = failures[-1]
    if latest.get('failure_code') != 'grasp_failed':
        return True
    grasp_failures = sum(
        1 for item in failures
        if item.get('failure_code') == 'grasp_failed'
    )
    replans = sum(
        1 for plan in io.plans if plan.get('status') == 'replanned'
    )
    if grasp_failures > replans:
        return time.monotonic() - failed_at > min(timeout, 15.0)
    if replans >= grasp_failures:
        return False
    return False


def _launch_episode(
    episode, results_dir, planner_mode, vision_backend, timeout
):
    launch_file = 'vision_demo.launch.py'
    command = [
        'ros2', 'launch', 'agent_robot', launch_file,
        'image_path:={}'.format(episode['image_path']),
        'pybullet_connection_mode:=DIRECT',
        'planner_mode:={}'.format(planner_mode),
        'vision_backend:={}'.format(vision_backend),
        'output_dir:={}'.format(os.path.join(
            results_dir, episode['episode_id'], 'trajectory_runs'
        )),
    ]
    return subprocess.Popen(command)


def _run_episode(node, episode, results_dir, planner_mode,
                 vision_backend, timeout, recovery_probe=False):
    io = EpisodeIO(node)
    process = _launch_episode(
        episode, results_dir, planner_mode, vision_backend, timeout
    )
    started = time.monotonic()
    command_started = None
    attempt_finished = None
    status = None
    failure_type = None
    failed_at = time.monotonic()

    try:
        if not _spin_until(node, lambda: io.environment is not None, timeout):
            failure_type = 'perception_failure'
        else:
            # Let the planner and executor subscriptions settle after the
            # perception message arrives on a newly launched stack.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if recovery_probe:
                io.inject_grasp_failure(
                    episode['goal']['target_id']
                )
                rclpy.spin_once(node, timeout_sec=0.25)
            command_started = time.monotonic()
            io.publish_command(episode['instruction'])
            if not _spin_until(
                node,
                lambda: _episode_finished(io, failed_at, timeout),
                timeout,
            ):
                failure_type = 'execution_failure'
            else:
                status = _terminal_status(io.statuses)
                if status is None:
                    status = next(
                        (
                            item for item in reversed(io.statuses)
                            if item.get('status') == 'failed'
                        ),
                        None,
                    )
                if status and status.get('status') == 'rejected':
                    plan = io.plan or {}
                    reason = plan.get('failure_reason', '')
                    if (
                        plan.get('planner') == 'plan_validator'
                        or 'plan_validation_failed' in reason
                    ):
                        failure_type = 'invalid_action'
                    else:
                        failure_type = 'planning_failure'
                elif status and status.get('status') != 'succeeded':
                    failure_type = (
                        'invalid_action'
                        if 'invalid_action_schema'
                        in status.get('reason', '')
                        else 'execution_failure'
                    )
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        attempt_finished = time.monotonic()
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        io.close(node)

    wall_time = time.monotonic() - started
    execution_time = (
        attempt_finished - command_started
        if command_started is not None and attempt_finished is not None
        else None
    )
    success = status is not None and status.get('status') == 'succeeded'
    return {
        'episode_id': episode['episode_id'],
        'image': episode['image_path'],
        'instruction': episode['instruction'],
        'goal': episode['goal'],
        'scene_spec': episode['scene_spec_path'],
        'scene_graph': io.scene_graph,
        'perceived_environment': io.environment,
        'plan': io.plan,
        'plan_history': io.plans,
        'trajectory': io.trajectory,
        'trajectory_length': len(io.trajectory),
        'failures': [
            item for item in io.statuses
            if item.get('status') == 'failed'
        ],
        'replans': max(
            [
                plan.get('replans', 0)
                for plan in io.plans
                if isinstance(plan.get('replans'), int)
            ] or [0]
        ),
        'success': success,
        'recovered': bool(
            success
            and any(
                item.get('failure_code') == 'grasp_failed'
                for item in io.statuses
            )
            and any(
                plan.get('status') == 'replanned'
                for plan in io.plans
            )
        ),
        'failure_type': None if success else failure_type,
        'failure_reason': (
            status.get('reason', '') if status is not None else ''
        ),
        'execution_time_seconds': (
            round(execution_time, 4)
            if execution_time is not None else None
        ),
        'episode_wall_time_seconds': round(wall_time, 4),
        'planner_mode': planner_mode,
        'vision_backend': vision_backend,
    }


def load_episodes(input_dir):
    """Read task labels produced by ``generate_benchmark_episodes``."""
    episodes = []
    for name in sorted(os.listdir(input_dir)):
        episode_dir = os.path.join(input_dir, name)
        task_path = os.path.join(episode_dir, 'task.json')
        if not os.path.isdir(episode_dir) or not os.path.isfile(task_path):
            continue
        with open(task_path, encoding='utf-8') as handle:
            task = json.load(handle)
        image_path = os.path.join(episode_dir, task['image'])
        if not os.path.isfile(image_path):
            raise ValueError('Missing episode image: {}'.format(image_path))
        episodes.append({
            'episode_id': task['episode_id'],
            'instruction': task['instruction'],
            'image_path': image_path,
            'goal': task.get('goal'),
            'scene_spec_path': os.path.join(
                episode_dir, task.get('scene_spec', 'scene_spec.json')
            ),
        })
    if not episodes:
        raise ValueError(
            'No generated episodes found in {}'.format(input_dir)
        )
    return episodes


def run_benchmark(
    input_dir,
    output_path,
    planner_mode='mock',
    vision_backend='synthetic_opencv',
    timeout=90.0,
    ros_domain_id=None,
):
    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    episode_results_dir = os.path.splitext(output_path)[0] + '_episodes'
    os.makedirs(episode_results_dir, exist_ok=True)
    episodes = load_episodes(input_dir)
    if ros_domain_id is None:
        ros_domain_id = secrets.randbelow(203) + 30
    if not 0 <= int(ros_domain_id) <= 232:
        raise ValueError('ROS domain ID must be between 0 and 232.')
    os.environ['ROS_DOMAIN_ID'] = str(int(ros_domain_id))
    rclpy.init()
    node = rclpy.create_node('m5_benchmark_runner')
    rows = []
    try:
        for index, episode in enumerate(episodes, start=1):
            node.get_logger().info(
                'Running episode {}/{}: {}'.format(
                    index, len(episodes), episode['episode_id']
                )
            )
            row = _run_episode(
                node,
                episode,
                os.path.dirname(output_path),
                planner_mode,
                vision_backend,
                timeout,
                recovery_probe=(index == 1),
            )
            rows.append(row)
            episode_output = os.path.join(
                episode_results_dir,
                episode['episode_id'] + '.json',
            )
            with open(episode_output, 'w', encoding='utf-8') as handle:
                json.dump(row, handle, ensure_ascii=False, indent=2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    results = {
        'schema_version': '1.0',
        'input_dir': input_dir,
        'planner_mode': planner_mode,
        'vision_backend': vision_backend,
        'ros_domain_id': int(ros_domain_id),
        'recovery_probe_episode': (
            episodes[0]['episode_id'] if episodes else None
        ),
        'episode_results_dir': episode_results_dir,
        'summary': benchmark_summary(rows),
        'episodes': rows,
    }
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    return results


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Run M5 image-to-execution benchmark episodes.'
    )
    parser.add_argument('--input-dir', default='datasets/m5_benchmark')
    parser.add_argument(
        '--output',
        default='experiments/final_results.json',
    )
    parser.add_argument(
        '--planner-mode', choices=('mock', 'deepseek'), default='mock'
    )
    parser.add_argument(
        '--vision-backend',
        choices=('opencv', 'synthetic_opencv', 'deepseek'),
        default='synthetic_opencv',
    )
    parser.add_argument('--timeout', type=float, default=90.0)
    parser.add_argument(
        '--ros-domain-id',
        default='auto',
        help='Isolated DDS domain ID 0-232, or auto.',
    )
    options = parser.parse_args(args=args)
    results = run_benchmark(
        options.input_dir,
        options.output,
        planner_mode=options.planner_mode,
        vision_backend=options.vision_backend,
        timeout=options.timeout,
        ros_domain_id=(
            None if options.ros_domain_id == 'auto'
            else int(options.ros_domain_id)
        ),
    )
    print(json.dumps(results['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
