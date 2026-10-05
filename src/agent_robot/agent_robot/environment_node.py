import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class EnvironmentNode(Node):

    def __init__(self):
        super().__init__('environment_node')

        self.publisher = self.create_publisher(
            String,
            '/environment_state',
            10
        )

        self.timer = self.create_timer(
            2.0,
            self.publish_environment
        )

        self.get_logger().info(
            'Environment node has started.'
        )

    def publish_environment(self):
        environment = {
            'robot_location': 'living_room',
            'objects': [
                {
                    'name': 'red_apple',
                    'color': 'red',
                    'location': 'table'
                },
                {
                    'name': 'basket',
                    'color': 'brown',
                    'location': 'floor'
                },
                {
                    'name': 'cup',
                    'color': 'blue',
                    'location': 'desk'
                }
            ]
        }

        message = String()
        message.data = json.dumps(
            environment,
            ensure_ascii=False
        )

        self.publisher.publish(message)

        self.get_logger().debug(
            f'Published environment: {message.data}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = EnvironmentNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()