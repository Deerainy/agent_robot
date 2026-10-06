"""Trajectory recorder node: latched observer of the execution streams.

M3: assembles at most one active Run at a time from ``/task_plan``,
``/task_status`` and ``/scene_graph`` and writes one JSON file per run
(atomic tmp + rename). Pure observer: publishes nothing; the task
executor remains the single authority for action validation and state.
"""

import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from agent_robot.trajectory import RunBuilder

# Must match task_executor's publisher profile: the recorder subscribes
# to the latched authoritative scene graph (same as task_planner).
SCENE_GRAPH_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class TrajectoryRecorder(Node):

    def __init__(self):
        super().__init__('trajectory_recorder')

        self.declare_parameter('output_dir', '~/ros2_ws/trajectories')
        raw_dir = self.get_parameter('output_dir').value
        self.output_dir = os.path.expanduser(raw_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.builder = RunBuilder()

        self.create_subscription(
            String, '/task_plan', self.plan_callback, 10
        )
        self.create_subscription(
            String, '/task_status', self.status_callback, 10
        )
        self.create_subscription(
            String, '/scene_graph', self.graph_callback, SCENE_GRAPH_QOS
        )

        # Drains terminal runs after the final-graph wait and idle-timeout
        # flushes; short period so the 0.5s terminal delay is precise.
        self.flush_timer = self.create_timer(0.25, self.flush_ready)

        self.get_logger().info(
            'Trajectory recorder started; output_dir={}'.format(
                self.output_dir
            )
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def plan_callback(self, msg):
        try:
            self.builder.on_plan(json.loads(msg.data))
        except json.JSONDecodeError as error:
            self.get_logger().error('Invalid plan JSON: {}'.format(error))

    def status_callback(self, msg):
        try:
            self.builder.on_status(json.loads(msg.data))
        except json.JSONDecodeError as error:
            self.get_logger().error('Invalid status JSON: {}'.format(error))

    def graph_callback(self, msg):
        try:
            self.builder.on_scene_graph(json.loads(msg.data))
        except json.JSONDecodeError as error:
            self.get_logger().error(
                'Invalid scene graph JSON: {}'.format(error)
            )

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def flush_ready(self):
        for run in self.builder.pop_ready_runs():
            self.write_run(run)

    def write_run(self, run):
        path = os.path.join(
            self.output_dir, '{}.json'.format(run['run_id'])
        )
        tmp_path = path + '.tmp'

        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(run, handle, ensure_ascii=False, indent=2)

        os.replace(tmp_path, path)

        self.get_logger().info(
            'Run flushed: {} outcome={} command={}'.format(
                run['run_id'], run['outcome'], run['command']
            )
        )

    def destroy_node(self):
        active = self.builder.close_active('recorder_shutdown')

        if active is not None:
            self.write_run(active)

        for run in self.builder.pop_ready_runs():
            self.write_run(run)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
