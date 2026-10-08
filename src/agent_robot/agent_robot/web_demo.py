"""Small Gradio frontend for the existing ROS 2 vision demo."""

import glob
import json
import os
import random
import shutil
import signal
import subprocess
import time
from datetime import datetime


WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)


class EpisodeCollector:
    """Collect existing perception, planning and execution ROS topics."""

    def __init__(self, node):
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from std_msgs.msg import String

        from agent_robot.ros_qos import ENVIRONMENT_STATE_QOS

        scene_graph_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.perception = None
        self.environment = None
        self.scene_graph = None
        self.plan = None
        self.plans = []
        self.statuses = []
        self.trajectory = []
        self.command_publisher = node.create_publisher(
            String, '/user_command', 10
        )
        self._subscriptions = [
            node.create_subscription(
                String, '/perception_result', self._perception, 10
            ),
            node.create_subscription(
                String,
                '/environment_state',
                self._environment,
                ENVIRONMENT_STATE_QOS,
            ),
            node.create_subscription(
                String,
                '/perceived_scene_graph',
                self._perceived_scene_graph,
                10,
            ),
            node.create_subscription(
                String, '/scene_graph', self._scene_graph, scene_graph_qos
            ),
            node.create_subscription(
                String, '/task_plan', self._plan, 10
            ),
            node.create_subscription(
                String, '/task_status', self._status, 10
            ),
            node.create_subscription(
                String, '/trajectory_point', self._trajectory_point, 100
            ),
        ]

    def publish_command(self, command):
        from std_msgs.msg import String

        message = String()
        message.data = command
        self.command_publisher.publish(message)

    def close(self, node):
        for subscription in self._subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_publisher(self.command_publisher)

    @staticmethod
    def _decode(message):
        return json.loads(message.data)

    def _perception(self, message):
        self.perception = self._decode(message)

    def _environment(self, message):
        payload = self._decode(message)
        if payload.get('source') == 'perception':
            self.environment = payload

    def _perceived_scene_graph(self, message):
        self.scene_graph = self._decode(message)

    def _scene_graph(self, message):
        graph = self._decode(message)
        if graph.get('source') == 'execution' or self.scene_graph is None:
            self.scene_graph = graph

    def _plan(self, message):
        self.plan = self._decode(message)
        self.plans.append(self.plan)

    def _status(self, message):
        self.statuses.append(self._decode(message))

    def _trajectory_point(self, message):
        self.trajectory.append(self._decode(message))

    def is_finished(self, failure_started, recovery_wait):
        if not self.statuses:
            return False
        latest = self.statuses[-1]
        if latest.get('status') in ('succeeded', 'rejected'):
            return True
        if latest.get('status') != 'failed':
            return False
        failures = [
            item for item in self.statuses
            if item.get('status') == 'failed'
        ]
        if latest.get('failure_code') != 'grasp_failed':
            return True
        replans = sum(
            1 for plan in self.plans
            if plan.get('status') == 'replanned'
        )
        if replans >= len(failures):
            return False
        if failure_started is None:
            return False
        return time.monotonic() - failure_started >= recovery_wait


def _spin_until(rclpy, node, predicate, timeout, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
        if process is not None and process.poll() is not None:
            return False
    return predicate()


def _run_episode(image_path, command, mode, connection_mode):
    import rclpy

    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise ValueError('Uploaded image does not exist.')

    run_id = datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    run_dir = os.path.join(WORKSPACE_ROOT, 'experiments', 'web_demo', run_id)
    os.makedirs(run_dir, exist_ok=False)
    local_image = os.path.join(
        run_dir, 'input' + (os.path.splitext(image_path)[1] or '.png')
    )
    shutil.copy2(image_path, local_image)
    output_dir = os.path.join(run_dir, 'trajectories')
    results_dir = os.path.join(run_dir, 'scene_graphs')
    log_path = os.path.join(run_dir, 'ros_launch.log')
    planner_mode, vision_backend = {
        'Local / mock': ('mock', 'opencv'),
        'Real DeepSeek': ('deepseek', 'deepseek'),
    }[mode]

    old_domain = os.environ.get('ROS_DOMAIN_ID')
    domain_id = str(random.randint(30, 232))
    os.environ['ROS_DOMAIN_ID'] = domain_id
    node = None
    collector = None
    process = None
    log_handle = None
    try:
        rclpy.init(args=[])
        node = rclpy.create_node('embodiedplan_web_demo')
        collector = EpisodeCollector(node)
        log_handle = open(log_path, 'w', encoding='utf-8')
        process = subprocess.Popen(
            [
                'ros2', 'launch', 'agent_robot', 'vision_demo.launch.py',
                'image_path:={}'.format(local_image),
                'results_dir:={}'.format(results_dir),
                'output_dir:={}'.format(output_dir),
                'pybullet_connection_mode:={}'.format(connection_mode),
                'planner_mode:={}'.format(planner_mode),
                'vision_backend:={}'.format(vision_backend),
            ],
            cwd=WORKSPACE_ROOT,
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        if not _spin_until(
            rclpy,
            node,
            lambda: collector.environment is not None,
            timeout=120.0,
            process=process,
        ):
            raise RuntimeError(
                'Image perception did not produce an environment. '
                'Check the ROS launch log.'
            )

        # Let planner and executor subscriptions consume the latched scene.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        collector.publish_command(command)

        failure_started = None
        observed_failures = 0
        execution_deadline = time.monotonic() + 180.0
        while time.monotonic() < execution_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            failure_count = sum(
                1 for status in collector.statuses
                if status.get('status') == 'failed'
            )
            if failure_count > observed_failures:
                observed_failures = failure_count
                failure_started = time.monotonic()
            if collector.is_finished(
                failure_started=failure_started,
                recovery_wait=12.0,
            ):
                break
            if process.poll() is not None:
                break
        else:
            raise TimeoutError(
                'ROS execution did not finish within 180 seconds.'
            )

        if not collector.statuses:
            raise RuntimeError(
                'The ROS stack exited without publishing task status.'
            )
        drain_deadline = time.monotonic() + 0.5
        while time.monotonic() < drain_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if collector is not None and node is not None:
            collector.close(node)
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if log_handle is not None:
            log_handle.close()
        if old_domain is None:
            os.environ.pop('ROS_DOMAIN_ID', None)
        else:
            os.environ['ROS_DOMAIN_ID'] = old_domain

    recorded_runs = []
    for path in sorted(glob.glob(os.path.join(output_dir, '*.json'))):
        with open(path, encoding='utf-8') as handle:
            recorded_runs.append(json.load(handle))
    recorded_run = recorded_runs[-1] if recorded_runs else None
    final_status = next(
        (
            status for status in reversed(collector.statuses)
            if status.get('status') in ('succeeded', 'rejected', 'failed')
        ),
        collector.statuses[-1],
    )
    objects = (
        collector.environment.get('objects', [])
        if collector.environment else []
    )
    failures = [
        status for status in collector.statuses
        if status.get('status') == 'failed'
    ]
    replans = [
        plan for plan in collector.plans
        if plan.get('status') == 'replanned'
    ]
    recovered = bool(
        final_status.get('status') == 'succeeded'
        and failures
        and replans
    )
    result = {
        'run_id': run_id,
        'image': local_image,
        'instruction': command,
        'mode': mode,
        'detected_objects': [
            {
                'id': obj.get('id'),
                'name': obj.get('name'),
                'type': obj.get('type'),
                'color': obj.get('color'),
                'confidence': obj.get('confidence'),
                'position': obj.get('position'),
                'bbox': obj.get('bbox'),
            }
            for obj in objects
        ],
        'scene_graph': collector.scene_graph,
        'plan': collector.plan,
        'plan_history': collector.plans,
        'execution_status': final_status,
        'trajectory_point_count': len(collector.trajectory),
        'failures': failures,
        'replans': len(replans),
        'recovered': recovered,
        'trajectory_recording': recorded_run,
        'run_directory': run_dir,
        'ros_log': log_path,
    }
    return result


def _render_summary(result):
    if result.get('error'):
        return '**任务未完成**\n\n{}'.format(result['error'])
    status = result.get('execution_status', {})
    objects = result.get('detected_objects', [])
    object_text = ', '.join(
        item.get('id') or item.get('name') or item.get('type', 'object')
        for item in objects
    ) or '未检测到物体'
    plan = result.get('plan') or {}
    actions = plan.get('actions', [])
    action_text = '\n'.join(
        '{}. `{}`'.format(index, json.dumps(action, ensure_ascii=False))
        for index, action in enumerate(actions, start=1)
    ) or '无可执行动作'
    failures = result.get('failures') or []
    failure_text = '\n'.join(
        '- {} {}'.format(
            item.get('failure_code', 'failure'),
            item.get('reason', ''),
        )
        for item in failures
    ) or '无'
    return (
        '### 执行结果\n'
        '- 状态：**{}** ({})\n'
        '- 检测物体：{}\n'
        '- 轨迹点：{}\n'
        '- 重规划：{} 次；恢复成功：{}\n'
        '- 失败记录：\n{}\n\n'
        '### 动作计划\n{}'
    ).format(
        status.get('status', 'unknown'),
        status.get('reason', ''),
        object_text,
        result.get('trajectory_point_count', 0),
        result.get('replans', 0),
        '是' if result.get('recovered') else '否',
        failure_text,
        action_text,
    )


def run_from_ui(image_path, command, mode, show_gui):
    if not image_path:
        error = {'error': '请先上传图片。'}
        return _render_summary(error), error
    if not command or not command.strip():
        error = {'error': '请输入自然语言任务。'}
        return _render_summary(error), error
    try:
        result = _run_episode(
            image_path,
            command.strip(),
            mode,
            'GUI' if show_gui else 'DIRECT',
        )
        return _render_summary(result), result
    except Exception as error:
        result = {
            'error': '{}: {}'.format(type(error).__name__, error),
        }
        return _render_summary(result), result


def main():
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            'Gradio is not installed. Run: '
            "python3 -m pip install 'gradio<4'"
        ) from error

    with gr.Blocks(title='EmbodiedPlan Web Demo') as demo:
        gr.Markdown(
            '# EmbodiedPlan\n'
            '上传图片并输入任务，自动运行感知、规划、PyBullet 执行和轨迹记录。'
        )
        with gr.Row():
            image = gr.Image(
                label='场景图片',
                type='filepath',
                source='upload',
            )
            with gr.Column():
                command = gr.Textbox(
                    label='自然语言任务',
                    value='put apple into basket',
                    placeholder='例如：put apple into basket',
                )
                mode = gr.Radio(
                    ['Local / mock', 'Real DeepSeek'],
                    value='Local / mock',
                    label='运行模式',
                )
                show_gui = gr.Checkbox(
                    value=False,
                    label='单独打开 PyBullet GUI',
                )
                run_button = gr.Button('Run', variant='primary')
        summary = gr.Markdown()
        details = gr.JSON(label='完整 episode 结果')
        run_button.click(
            run_from_ui,
            inputs=[image, command, mode, show_gui],
            outputs=[summary, details],
        )

    demo.queue(concurrency_count=1).launch(
        server_name='127.0.0.1',
        server_port=7860,
        share=False,
    )


if __name__ == '__main__':
    main()
