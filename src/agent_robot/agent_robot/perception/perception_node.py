"""ROS 2 node publishing structured detections and mapped scene objects."""

import json
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from agent_robot.perception.coordinate_mapper import CoordinateMapper
from agent_robot.perception.pipeline import perceive_image
from agent_robot.ros_qos import ENVIRONMENT_STATE_QOS


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        self.declare_parameter('world_x_min', 0.38)
        self.declare_parameter('world_x_max', 0.76)
        self.declare_parameter('world_y_min', -0.28)
        self.declare_parameter('world_y_max', 0.28)
        self.declare_parameter('roi_left', 0.0)
        self.declare_parameter('roi_top', 0.0)
        self.declare_parameter('roi_right', 1.0)
        self.declare_parameter('roi_bottom', 1.0)
        self.declare_parameter('results_dir', '')
        self.declare_parameter('image_path', '')
        self.declare_parameter(
            'vision_backend',
            os.environ.get('VISION_BACKEND', 'deepseek'),
        )
        self._startup_image = self.get_parameter('image_path').value
        self.vision_backend = self.get_parameter('vision_backend').value
        self.mapper = CoordinateMapper(
            world_x=(
                self.get_parameter('world_x_min').value,
                self.get_parameter('world_x_max').value,
            ),
            world_y=(
                self.get_parameter('world_y_min').value,
                self.get_parameter('world_y_max').value,
            ),
            image_roi=(
                self.get_parameter('roi_left').value,
                self.get_parameter('roi_top').value,
                self.get_parameter('roi_right').value,
                self.get_parameter('roi_bottom').value,
            ),
        )
        self.results_dir = os.path.expanduser(
            self.get_parameter('results_dir').value
        )
        self.image_subscription = self.create_subscription(
            String,
            '/image_path',
            self.image_callback,
            10,
        )
        self.perception_publisher = self.create_publisher(
            String,
            '/perception_result',
            10,
        )
        self.environment_publisher = self.create_publisher(
            String,
            '/environment_state',
            ENVIRONMENT_STATE_QOS,
        )
        self.scene_graph_publisher = self.create_publisher(
            String,
            '/perceived_scene_graph',
            10,
        )
        self.get_logger().info(
            '{} perception ready; waiting for /image_path.'.format(
                self.vision_backend
            )
        )
        self._startup_timer = self.create_timer(
            0.5, self._process_startup_image
        )

    def _process_startup_image(self):
        self._startup_timer.cancel()
        if self._startup_image:
            message = String()
            message.data = self._startup_image
            self.image_callback(message)

    def image_callback(self, msg):
        image_path = os.path.abspath(os.path.expanduser(msg.data.strip()))
        try:
            perception, environment, graph = perceive_image(
                image_path,
                self.mapper,
                vision_backend=self.vision_backend,
            )
            self._publish(self.perception_publisher, perception)
            self._publish(self.environment_publisher, environment)
            self._publish(self.scene_graph_publisher, graph.to_dict())
            placement = environment.get('simulation_placement', {})
            for warning in placement.get('warnings', []):
                self.get_logger().warning(
                    'Perception/placement warning: {}'.format(warning)
                )
            for adjustment in placement.get('adjustments', []):
                self.get_logger().warning(
                    'Simulation-only position correction for {}: {} -> {}'
                    .format(
                        adjustment['object'],
                        adjustment['perception_position'],
                        adjustment['simulation_position'],
                    )
                )
            if self.results_dir:
                self._write_result(
                    perception['scenario_id'], graph.to_dict()
                )
            self.get_logger().info(
                'Perceived {} objects from {}.'.format(
                    len(environment['objects']), image_path
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.get_logger().error(
                'Image perception failed for {}: {}'.format(
                    image_path, error
                )
            )

    @staticmethod
    def _publish(publisher, payload):
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(message)

    def _write_result(self, scenario_id, graph):
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(
            self.results_dir,
            '{}_scene_graph.json'.format(scenario_id),
        )
        temporary_path = path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as output_file:
            json.dump(graph, output_file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
