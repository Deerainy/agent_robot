import json

import pybullet as p
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from agent_robot.pybullet_robot import (
    create_scene,
    run_pick,
    run_pick_and_place,
)


class PyBulletBridge(Node):

    def __init__(self):
        super().__init__('pybullet_bridge')

        self.robot_id, self.apple_id = create_scene()
        self.executing = False
        self.last_command = None

        self.plan_subscription = self.create_subscription(
            String,
            '/task_plan',
            self.plan_callback,
            10
        )

        # 无任务时也持续刷新物理仿真
        self.simulation_timer = self.create_timer(
            1.0 / 240.0,
            self.step_simulation
        )

        self.get_logger().info(
            'PyBullet Bridge started. Waiting for /task_plan.'
        )

    def step_simulation(self):
        if not self.executing and p.isConnected():
            p.stepSimulation()

    def plan_callback(self, msg):
        if self.executing:
            self.get_logger().warning(
                'Robot is executing another task.'
            )
            return

        try:
            plan = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f'Invalid task plan: {error}'
            )
            return

        command = plan.get('command', '')
        feasible = plan.get('feasible', False)
        status = plan.get('status', '')

        if not feasible or status == 'rejected':
            self.get_logger().warning(
                f'Ignoring rejected task: {plan.get("reason", "")}'
            )
            return

        # 防止同一任务被重复执行
        if command == self.last_command:
            self.get_logger().warning(
                'Duplicate task ignored.'
            )
            return

        command_text = command.lower()

        has_apple = (
            'apple' in command_text
            or '苹果' in command_text
        )
        has_basket = (
            'basket' in command_text
            or '篮子' in command_text
            or '篮筐' in command_text
        )


        self.executing = True
        self.last_command = command

        self.get_logger().info(
            f'Executing task in PyBullet: {command}'
        )

        try:
            if has_apple and has_basket:
                self.get_logger().info(
                    'Selected robot skill: pick_and_place'
                )

                run_pick_and_place(
                    self.robot_id,
                    self.apple_id
                )

            elif has_apple:
                self.get_logger().info(
                    'Selected robot skill: pick'
                )

                run_pick(
                    self.robot_id,
                    self.apple_id
                )

            else:
                self.get_logger().warning(
                    f'No robot skill is available for: {command}'
                )
                return

            self.get_logger().info(
                'PyBullet task completed successfully.'
            )

        except Exception as error:
            self.get_logger().error(
                f'PyBullet execution failed: {error}'
            )

        finally:
            self.executing = False


def main(args=None):
    rclpy.init(args=args)
    node = PyBulletBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()