import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String


class SceneLoader(Node):

    def __init__(self):
        super().__init__('scene_loader')

        self.publisher = self.create_publisher(
            String,
            '/image_path',
            10
        )

        package_directory = get_package_share_directory('agent_robot')

        self.image_path = os.path.join(
            package_directory,
            'images',
            'scene.png'
        )

        self.image_published = False

        # 等待其他节点启动后再发布图片路径
        self.timer = self.create_timer(
            2.0,
            self.publish_image_path
        )

        self.get_logger().info(
            'Scene Loader started. Waiting to publish the default image.'
        )

    def publish_image_path(self):
        if self.image_published:
            return

        if not os.path.isfile(self.image_path):
            self.get_logger().error(
                f'Default scene image not found: {self.image_path}'
            )
            self.timer.cancel()
            return

        message = String()
        message.data = self.image_path
        self.publisher.publish(message)

        self.get_logger().info(
            f'Published default image path: {self.image_path}'
        )

        self.image_published = True
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)

    node = SceneLoader()

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