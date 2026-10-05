import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskExecutor(Node):

    def __init__(self):
        super().__init__('task_executor')

        self.subscription = self.create_subscription(
            String,
            '/task_plan',
            self.plan_callback,
            10
        )

        self.status_publisher = self.create_publisher(
            String,
            '/task_status',
            10
        )

        self.get_logger().info('Task Executor node has started.')

    def plan_callback(self, msg):
        try:
            plan = json.loads(msg.data)
            command = plan['command']
            steps = plan['steps']

            self.get_logger().info(f'Received task: {command}')

            for index, step in enumerate(steps, start=1):
                self.get_logger().info(
                    f'Executing step {index}/{len(steps)}: {step}'
                )

                status = {
                    'command': command,
                    'current_step': index,
                    'total_steps': len(steps),
                    'action': step,
                    'status': 'executing'
                }

                status_msg = String()
                status_msg.data = json.dumps(status, ensure_ascii=False)
                self.status_publisher.publish(status_msg)

                time.sleep(1)

            completed_status = {
                'command': command,
                'status': 'completed'
            }

            completed_msg = String()
            completed_msg.data = json.dumps(
                completed_status,
                ensure_ascii=False
            )
            self.status_publisher.publish(completed_msg)

            self.get_logger().info('Task completed.')

        except (json.JSONDecodeError, KeyError) as error:
            self.get_logger().error(f'Invalid task plan: {error}')


def main(args=None):
    rclpy.init(args=args)

    node = TaskExecutor()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()