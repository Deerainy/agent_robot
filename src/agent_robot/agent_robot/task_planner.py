import json
import os
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskPlanner(Node):

    def __init__(self):
        super().__init__('task_planner')

        self.subscription = self.create_subscription(
            String,
            '/user_command',
            self.command_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/task_plan',
            10
        )

        self.get_logger().info(
            'DeepSeek Task Planner node has started.'
        )

    def command_callback(self, msg):
        command = msg.data
        self.get_logger().info(f'Received command: {command}')
        self.get_logger().info('Generating plan with DeepSeek...')

        plan = self.generate_plan(command)

        output_msg = String()
        output_msg.data = json.dumps(plan, ensure_ascii=False)
        self.publisher.publish(output_msg)

        self.get_logger().info(f'Published plan: {output_msg.data}')

    def generate_plan(self, command):
        api_key = os.environ.get('DEEPSEEK_API_KEY')

        if not api_key:
            self.get_logger().warning(
                'DEEPSEEK_API_KEY is not configured. '
                'Using fallback plan.'
            )
            return self.fallback_plan(command)

        prompt = f"""
请将下面的用户命令拆分成机器人能够执行的动作步骤。

用户命令：{command}

要求：
1. 生成3到8个步骤。
2. 每个步骤只描述一个明确动作。
3. 步骤必须按照执行顺序排列。
4. 不要输出解释。
5. 严格输出下面格式的JSON：

{{
    "steps": [
        "步骤1",
        "步骤2",
        "步骤3"
    ]
}}
"""

        request_data = {
            'model': 'deepseek-flash',
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        '你是具身机器人任务规划器，负责将自然语言'
                        '任务转换成可执行的机器人动作序列。'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'response_format': {
                'type': 'json_object'
            },
            'stream': False
        }

        request = urllib.request.Request(
            'https://api.deepseek.com/chat/completions',
            data=json.dumps(request_data).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=60
            ) as response:
                result = json.loads(
                    response.read().decode('utf-8')
                )

            content = result['choices'][0]['message']['content']
            generated_plan = json.loads(content)
            steps = generated_plan['steps']

            if not isinstance(steps, list) or not steps:
                raise ValueError('DeepSeek returned an empty plan.')

            return {
                'command': command,
                'steps': steps,
                'planner': 'deepseek',
                'status': 'planned'
            }

        except urllib.error.HTTPError as error:
            error_message = error.read().decode('utf-8')
            self.get_logger().error(
                f'DeepSeek HTTP error {error.code}: '
                f'{error_message}'
            )

        except Exception as error:
            self.get_logger().error(
                f'DeepSeek planning failed: {error}'
            )

        self.get_logger().warning('Using fallback plan.')
        return self.fallback_plan(command)

    def fallback_plan(self, command):
        return {
            'command': command,
            'steps': [
                '分析用户任务',
                '感知当前环境',
                '确定目标对象的位置',
                '执行相应机器人动作',
                '检查任务执行结果'
            ],
            'planner': 'fallback',
            'status': 'planned'
        }


def main(args=None):
    rclpy.init(args=args)

    node = TaskPlanner()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()