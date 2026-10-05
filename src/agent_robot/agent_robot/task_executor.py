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

        self.failure_keyword = None

        # 失败模拟订阅
        self.failure_subscription = self.create_subscription(
            String,
            '/simulate_failure',
            self.failure_callback,
            10
        )

        self.get_logger().info('Task Executor node has started.')

    def failure_callback(self, msg):
        self.failure_keyword = msg.data.strip()

        self.get_logger().warning(
            f'Next action containing "{self.failure_keyword}" '
            'will be simulated as failed.'
        )

    def plan_callback(self, msg):
        try:
            plan = json.loads(msg.data)
            command = plan['command']
            steps = plan.get('steps', [])
            feasible = plan.get('feasible', True)
            reason = plan.get('reason', '')

            self.get_logger().info(f'Received task: {command}')

            if not feasible:
                self.get_logger().warning(
                    f'Task rejected: {reason}'
                )

                rejected_status = {
                    'command': command,
                    'status': 'rejected',
                    'reason': reason
                }

                rejected_msg = String()
                rejected_msg.data = json.dumps(
                    rejected_status,
                    ensure_ascii=False
                )
                self.status_publisher.publish(rejected_msg)
                return

            for index, step in enumerate(steps, start=1):

                if (
                    self.failure_keyword
                    and self.failure_keyword in step
                ):
                    failure_reason = (
                        f'动作“{step}”执行失败：检测到模拟障碍。'
                    )

                    self.get_logger().error(failure_reason)

                    failure_status = {
                        'command': command,
                        'status': 'failed',
                        'failed_step': step,
                        'step_index': index,
                        'reason': failure_reason
                    }

                    failure_msg = String()
                    failure_msg.data = json.dumps(
                        failure_status,
                        ensure_ascii=False
                    )
                    self.status_publisher.publish(failure_msg)

                    # 只让这次动作失败，避免以后一直失败
                    self.failure_keyword = None
                    return

                self.get_logger().info(
                    f'Executing step {index}/{len(steps)}: {step}'
                )

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