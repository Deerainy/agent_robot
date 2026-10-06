import json
import os
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from agent_robot.skill_registry import (
    ActionSchemaError,
    parse_actions,
    skill_catalog_text,
)

SCHEMA_VERSION = '2.0'


class TaskPlanner(Node):

    def __init__(self):
        super().__init__('task_planner')

        self.command_subscription = self.create_subscription(
            String,
            '/user_command',
            self.command_callback,
            10
        )

        self.environment_subscription = self.create_subscription(
            String,
            '/environment_state',
            self.environment_callback,
            10
        )

        self.status_subscription = self.create_subscription(
            String,
            '/task_status',
            self.status_callback,
            10
        )

        self.plan_publisher = self.create_publisher(
            String,
            '/task_plan',
            10
        )

        self.current_environment = {
            'status': 'environment_not_received',
            'objects': []
        }
        self.environment_received = False
        self.replan_counts = {}

        self.get_logger().info('DeepSeek Task Planner node has started.')

    def environment_callback(self, msg):
        try:
            environment = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                'Invalid environment data: {}'.format(error)
            )
            return

        self.current_environment = environment

        if not self.environment_received:
            self.get_logger().info('Environment state received.')
            self.environment_received = True

    def status_callback(self, msg):
        try:
            status_data = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                'Invalid task status: {}'.format(error)
            )
            return

        if status_data.get('status') != 'failed':
            return

        command = status_data.get('command', '')
        failed_action = status_data.get('action')
        reason = status_data.get('reason', 'unknown reason')

        replan_count = self.replan_counts.get(command, 0) + 1
        self.replan_counts[command] = replan_count

        self.get_logger().warning(
            'Execution failed at action {}: {}'.format(
                failed_action, reason
            )
        )
        self.get_logger().info(
            'Generating a revised plan with DeepSeek '
            '(replans={})...'.format(replan_count)
        )

        failure_context = {
            'failed_action': failed_action,
            'reason': reason,
            'replans': replan_count
        }

        revised_plan = self.generate_plan(
            command,
            failure_context=failure_context
        )
        revised_plan['command'] = command

        if revised_plan.get('feasible'):
            revised_plan['status'] = 'replanned'

        message = String()
        message.data = json.dumps(
            revised_plan,
            ensure_ascii=False
        )
        self.plan_publisher.publish(message)

        self.get_logger().info(
            'Published revised plan: {}'.format(message.data)
        )

    def command_callback(self, msg):
        command = msg.data
        self.get_logger().info('Received command: {}'.format(command))
        self.get_logger().info('Generating plan with DeepSeek...')

        plan = self.generate_plan(command)

        message = String()
        message.data = json.dumps(plan, ensure_ascii=False)
        self.plan_publisher.publish(message)

        self.get_logger().info('Published plan: {}'.format(message.data))

    def generate_plan(self, command, failure_context=None):
        api_key = os.environ.get('DEEPSEEK_API_KEY')

        if not api_key:
            self.get_logger().warning(
                'DEEPSEEK_API_KEY is not configured. '
                'Using fallback rejection.'
            )
            return self.fallback_plan(command)

        environment_text = json.dumps(
            self.current_environment,
            ensure_ascii=False,
            indent=2
        )

        object_names = self._environment_object_names()
        object_text = ', '.join(object_names) or '(environment is empty)'

        if failure_context:
            failure_text = json.dumps(
                {
                    'failed_action': failure_context.get('failed_action'),
                    'reason': failure_context.get('reason')
                },
                ensure_ascii=False,
                indent=2
            )
            replans = failure_context.get('replans', 1)
        else:
            failure_text = '无，这是首次规划。'
            replans = 0

        prompt = """
你是具身机器人的任务规划器，必须把自然语言任务编译成结构化机器人技能序列。

用户命令：
{command}

机器人当前感知到的环境：
{environment_text}

环境中可引用的物体名称（object/target 只能取这些英文名）：
{object_text}

{skill_catalog}

此前执行失败信息（若有，必须针对失败动作和原因调整，避免原样重试）：
{failure_text}

规划规则：
1. 用户明确要求操作的物体与目标容器必须出现在环境物体名称中，严禁虚构。
2. actions 中的 skill 只能使用上面目录列出的 3 个技能；严禁输出
   open_gripper、close_gripper 等内部原语或任何其他动词。
3. 每个动作只引用一个物体；place 必须同时给出 object 与 target。
4. 典型顺序为 move_to(物体) -> pick(物体) -> move_to(目标) ->
   place(物体, 目标)；根据任务需要生成 2 到 6 个动作，不得有冗余动作。
5. 缺少必要物体或目标时，feasible 必须为 false，actions 必须为空数组。
6. 只输出 JSON，不要输出 JSON 之外的任何解释或 Markdown 代码块。

可执行时严格输出：
{{
    "feasible": true,
    "reason": "",
    "actions": [
        {{"skill": "move_to", "object": "物体英文名"}},
        {{"skill": "pick", "object": "物体英文名"}},
        {{"skill": "move_to", "object": "目标英文名"}},
        {{"skill": "place", "object": "物体英文名", "target": "目标英文名"}}
    ]
}}

不可执行时严格输出：
{{
    "feasible": false,
    "reason": "不可执行的具体原因",
    "actions": []
}}
""".format(
            command=command,
            environment_text=environment_text,
            object_text=object_text,
            skill_catalog=skill_catalog_text(),
            failure_text=failure_text
        )

        request_data = {
            'model': 'deepseek-flash',
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        '你是具身机器人任务规划器，负责把自然语言任务'
                        '编译成结构化机器人技能 JSON。'
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
                'Authorization': 'Bearer {}'.format(api_key)
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))

            content = result['choices'][0]['message']['content']
            generated_plan = json.loads(content)

            feasible = generated_plan.get('feasible', False)
            reason = generated_plan.get('reason', '')
            raw_actions = generated_plan.get('actions', [])

            if not isinstance(feasible, bool):
                raise ValueError('Invalid feasible value.')

            if not isinstance(raw_actions, list):
                raise ValueError('Invalid actions value.')

            if feasible:
                actions = [
                    action.to_dict()
                    for action in parse_actions(raw_actions)
                ]
                status = 'planned'
            else:
                if raw_actions:
                    raise ValueError(
                        'Infeasible plan must contain empty actions.'
                    )
                actions = []
                status = 'rejected'

            return {
                'command': command,
                'schema_version': SCHEMA_VERSION,
                'feasible': feasible,
                'reason': reason,
                'actions': actions,
                'planner': 'deepseek',
                'status': status,
                'replans': replans
            }

        except ActionSchemaError as error:
            self.get_logger().error(
                'Generated actions violate the skill protocol: {}'.format(
                    error
                )
            )
        except urllib.error.HTTPError as error:
            error_message = error.read().decode('utf-8')
            self.get_logger().error(
                'DeepSeek HTTP error {}: {}'.format(
                    error.code, error_message
                )
            )
        except Exception as error:
            self.get_logger().error(
                'DeepSeek planning failed: {}'.format(error)
            )

        self.get_logger().warning('Using fallback rejection.')
        return self.fallback_plan(command, replans=replans)

    def _environment_object_names(self):
        names = []

        for obj in self.current_environment.get('objects', []):
            if isinstance(obj, dict):
                name = obj.get('name')
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())

        return names

    def fallback_plan(self, command, replans=0):
        return {
            'command': command,
            'schema_version': SCHEMA_VERSION,
            'feasible': False,
            'reason': '任务规划服务暂时不可用或输出不合法，无法安全生成计划。',
            'actions': [],
            'planner': 'fallback',
            'status': 'rejected',
            'replans': replans
        }


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
