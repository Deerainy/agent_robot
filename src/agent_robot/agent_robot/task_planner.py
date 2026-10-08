import json
import os
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from agent_robot.plan_validator import (
    build_validation_feedback,
    evaluate_generated_plan,
)
from agent_robot.ros_qos import ENVIRONMENT_STATE_QOS
from agent_robot.scene_graph import SceneGraph
from agent_robot.skill_registry import skill_catalog_text

SCHEMA_VERSION = '2.1'

# Must match task_executor's publisher profile: the planner subscribes to the
# latched authoritative scene graph.
SCENE_GRAPH_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class TaskPlanner(Node):

    def __init__(self):
        super().__init__('task_planner')
        self.declare_parameter('planner_mode', 'deepseek')
        self.declare_parameter(
            'api_url',
            os.environ.get(
                'DEEPSEEK_API_URL',
                'https://api.deepseek.com/chat/completions',
            ),
        )
        self.declare_parameter('api_key_env', 'DEEPSEEK_API_KEY')
        self.declare_parameter(
            'model',
            os.environ.get('DEEPSEEK_CHAT_MODEL', 'deepseek-flash'),
        )
        self.planner_mode = self.get_parameter('planner_mode').value
        self.api_url = self.get_parameter('api_url').value
        self.api_key_env = self.get_parameter('api_key_env').value
        self.model = self.get_parameter('model').value

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
            ENVIRONMENT_STATE_QOS
        )

        self.scene_graph_subscription = self.create_subscription(
            String,
            '/scene_graph',
            self.scene_graph_callback,
            SCENE_GRAPH_QOS
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
        self.scene_graph_received = False
        self.latest_scene_graph = None
        self.replan_counts = {}
        self.pending_command = None
        self.pending_command_timer = None

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

        if (
            environment.get('source') == 'perception'
            and self.pending_command is not None
        ):
            command = self.pending_command
            self.pending_command = None
            self._clear_pending_command_timer()
            self._process_command(command)

    def scene_graph_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_scene_graph = SceneGraph.from_dict(data)
        except (json.JSONDecodeError, ValueError) as error:
            self.get_logger().error(
                'Invalid scene graph data: {}'.format(error)
            )
            return

        if not self.scene_graph_received:
            self.get_logger().info(
                'Scene graph received from executor (source={}).'.format(
                    self.latest_scene_graph.source
                )
            )
            self.scene_graph_received = True

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

        if status_data.get('failure_code') != 'grasp_failed':
            return

        command = status_data.get('command', '')
        failed_action = status_data.get('action')
        reason = status_data.get('reason', 'unknown reason')
        action_data = failed_action if isinstance(failed_action, dict) else {}
        if action_data.get('skill') != 'pick':
            return

        replan_count = self.replan_counts.get(command, 0)
        if replan_count >= 1:
            self.get_logger().error(
                'grasp_failed recovery limit reached for command: {}'.format(
                    command
                )
            )
            return
        replan_count += 1
        self.replan_counts[command] = replan_count

        self.get_logger().warning(
            'Execution failed at action {}: {}'.format(
                failed_action, reason
            )
        )
        self.get_logger().info(
            'Generating one revised plan with {} (replans={})...'.format(
                self.planner_mode, replan_count
            )
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
        revised_plan['replans'] = replan_count

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
        command = msg.data.strip()
        if not command:
            self.get_logger().error('Ignoring an empty user command.')
            return

        if self.current_environment.get('source') != 'perception':
            if self.pending_command is not None:
                self._publish_plan(self._rejected_plan(
                    command,
                    '已有任务正在等待图片感知场景。',
                    planner='scene_validator',
                ))
                return
            self.pending_command = command
            self.pending_command_timer = self.create_timer(
                90.0, self._expire_pending_command
            )
            self.get_logger().info(
                'Waiting for perception before planning the command.'
            )
            return

        self._process_command(command)

    def _process_command(self, command):
        self.replan_counts[command] = 0
        self.get_logger().info('Received command: {}'.format(command))
        graph = self._validation_graph()
        if graph is None or not graph.objects:
            self._publish_plan(self._rejected_plan(
                command,
                '尚未收到图片感知场景，不能安全规划。',
                planner='scene_validator',
            ))
            return
        if graph.source != 'perception':
            self._publish_plan(self._rejected_plan(
                command,
                '当前场景不是图片感知结果，拒绝使用固定目录场景规划。',
                planner='scene_validator',
            ))
            return
        self.get_logger().info(
            'Generating a scene-grounded plan with {}.'.format(
                self.planner_mode
            )
        )

        plan = self.generate_plan(command)
        self._publish_plan(plan)

    def _expire_pending_command(self):
        command = self.pending_command
        if command is None:
            self._clear_pending_command_timer()
            return
        self.pending_command = None
        self._clear_pending_command_timer()
        self._publish_plan(self._rejected_plan(
            command,
            '等待图片感知场景超时，不能安全规划。',
            planner='scene_validator',
        ))

    def _clear_pending_command_timer(self):
        timer = self.pending_command_timer
        self.pending_command_timer = None
        if timer is not None:
            self.destroy_timer(timer)

    def _publish_plan(self, plan):
        message = String()
        message.data = json.dumps(plan, ensure_ascii=False)
        self.plan_publisher.publish(message)
        self.get_logger().info('Published plan: {}'.format(message.data))

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _validation_graph(self):
        """Choose the start graph for plan-time validation.

        Priority: latest executor graph (authoritative) -> a graph built
        from the current perception payload -> ``None`` (skip validation;
        catalog defaults must never masquerade as perception here).
        """
        if self.current_environment.get('source') == 'perception':
            perception_revision = self.current_environment.get(
                'world_revision', 0
            )
            if (
                self.latest_scene_graph is not None
                and self.latest_scene_graph.source == 'perception'
                and self.latest_scene_graph.scenario_id
                == self.current_environment.get('scenario_id')
                and self.latest_scene_graph.world_revision
                >= perception_revision
            ):
                return self.latest_scene_graph.clone()
            return SceneGraph.build_from_perception(
                self.current_environment
            )

        if (
            self.latest_scene_graph is not None
            and self.latest_scene_graph.source == 'perception'
        ):
            return self.latest_scene_graph.clone()

        if self.current_environment.get('objects'):
            return SceneGraph.build_from_environment(
                self.current_environment
            )

        return None

    def _grounding_names(self, graph):
        names = self._environment_object_names()

        if graph is not None:
            for name in graph.objects:
                if name not in names:
                    names.append(name)

        return names

    def generate_plan(self, command, failure_context=None):
        graph = self._validation_graph()
        if graph is None or graph.source != 'perception':
            return self._rejected_plan(
                command,
                '尚未收到有效的图片感知场景图。',
                planner='scene_validator',
            )

        if self.planner_mode == 'mock':
            return self._mock_plan(command, graph)
        if self.planner_mode != 'deepseek':
            return self._rejected_plan(
                command,
                'Unsupported planner_mode: {}'.format(self.planner_mode),
                planner='configuration',
            )

        api_key = os.environ.get(self.api_key_env)

        if not api_key:
            self.get_logger().warning(
                '{} is not configured; rejecting without execution.'.format(
                    self.api_key_env
                )
            )
            return self._rejected_plan(
                command,
                '{} is not configured; no plan was generated.'.format(
                    self.api_key_env
                ),
                planner='deepseek',
            )

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

        env_names = self._grounding_names(graph)
        object_text = ', '.join(env_names) or '(environment is empty)'
        scene_text = (
            graph.to_prompt_text()
            if graph is not None
            else '（暂无场景图：跳过序列级前置验证，执行器仍会逐动作验证）'
        )

        try:
            content = self._call_deepseek(
                self._build_prompt(
                    command,
                    object_text,
                    scene_text,
                    failure_text,
                    validation_text='无，这是首次生成。'
                )
            )
            generated_plan = json.loads(content)
        except urllib.error.HTTPError as error:
            error_message = error.read().decode('utf-8')
            self.get_logger().error(
                'DeepSeek HTTP error {}: {}'.format(
                    error.code, error_message
                )
            )
            return self.fallback_plan(command, replans=replans)
        except Exception as error:
            self.get_logger().error(
                'DeepSeek planning failed: {}'.format(error)
            )
            return self.fallback_plan(command, replans=replans)

        evaluation = evaluate_generated_plan(
            generated_plan, env_names, graph
        )

        # Exactly one self-repair attempt, and only when a graph exists to
        # validate the repaired output against.
        if evaluation.repairable and graph is not None:
            self.get_logger().warning(
                'Generated plan failed validation at {}: {}: {}. '
                'Requesting one self-repair.'.format(
                    evaluation.error_stage,
                    evaluation.error_code,
                    evaluation.error_detail
                )
            )

            validation_text = build_validation_feedback(evaluation, graph)

            try:
                repaired_content = self._call_deepseek(
                    self._build_prompt(
                        command,
                        object_text,
                        scene_text,
                        failure_text,
                        validation_text=validation_text
                    )
                )
                repaired_plan = json.loads(repaired_content)
            except urllib.error.HTTPError as error:
                error_message = error.read().decode('utf-8')
                self.get_logger().error(
                    'DeepSeek repair HTTP error {}: {}'.format(
                        error.code, error_message
                    )
                )
                return self.fallback_plan(command, replans=replans)
            except Exception as error:
                self.get_logger().error(
                    'DeepSeek repair failed: {}'.format(error)
                )
                return self.fallback_plan(command, replans=replans)

            evaluation = evaluate_generated_plan(
                repaired_plan, env_names, graph
            )

        if evaluation.feasible and evaluation.accepted:
            reachability_error = self._check_plan_reachability(
                evaluation.actions, graph
            )
            if reachability_error:
                return self._rejected_plan(
                    command,
                    'object_not_reachable: {}'.format(reachability_error),
                    replans=replans,
                    planner='scene_validator',
                )
            return {
                'command': command,
                'schema_version': SCHEMA_VERSION,
                'feasible': True,
                'reason': evaluation.reason,
                'failure_reason': '',
                'actions': evaluation.actions,
                'planner': 'deepseek',
                'status': 'planned',
                'replans': replans,
                'based_on_world_revision': graph.world_revision,
                'image_path': self.current_environment.get(
                    'image_path', ''
                ),
            }

        if evaluation.feasible:
            # The model produced something, but it is still unsafe after
            # the one allowed repair: reject without publishing bad actions.
            reason = 'plan_validation_failed: {}: {}{}'.format(
                evaluation.error_stage or 'validation',
                evaluation.error_code,
                evaluation.error_detail
            )
            self.get_logger().error(
                'Generated plan rejected by validation: {}'.format(reason)
            )
            return self._rejected_plan(
                command, reason, replans=replans, planner='plan_validator'
            )

        return self._rejected_plan(
            command,
            evaluation.reason,
            replans=replans,
            planner='deepseek'
        )

    def _call_deepseek(self, prompt):
        """Perform one chat-completion API call; return content string."""
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                '{} is not configured.'.format(self.api_key_env)
            )
        request_data = {
            'model': self.model,
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
            self.api_url,
            data=json.dumps(request_data).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer {}'.format(
                    os.environ.get('DEEPSEEK_API_KEY')
                )
            },
            method='POST'
        )
        request.add_header(
            'Authorization', 'Bearer {}'.format(api_key)
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))

        return result['choices'][0]['message']['content']

    def _build_prompt(
        self,
        command,
        object_text,
        scene_text,
        failure_text,
        validation_text
    ):
        return """
你是具身机器人的任务规划器，必须把自然语言任务编译成结构化机器人技能序列。

用户命令：
{command}

机器人当前感知到的环境：
{environment_text}

环境中可引用的物体名称（object/target 只能取这些英文名）：
{object_text}

{skill_catalog}

场景图（符号世界状态，动作序列必须满足前置条件）：
{scene_text}

此前执行失败信息（若有，必须针对失败动作和原因调整，避免原样重试）：
{failure_text}

上一版计划的自动验证反馈：
{validation_text}

规划规则：
1. 用户明确要求操作的物体与目标容器必须出现在环境物体名称中，严禁虚构。
2. actions 中的 skill 只能使用上面目录列出的 3 个技能；严禁输出
   open_gripper、close_gripper 等内部原语或任何其他动词。
3. 每个动作只引用一个物体；place 必须同时给出 object 与 target。
4. 技能前置条件（违反会被执行器拒绝，且不会产生任何运动）：
   - move_to(x) 仅要求 x 是场景中的可操作物体，语义为移动到 x 上方；
   - pick(x) 前必须先有针对同一 x 的 move_to(x)，x 必须可抓取，
     且夹爪必须为空；pick 内部自行完成张爪、下降、闭合与抬升；
   - place(x, t) 前必须正握持 x、且已有针对 t 的 move_to(t)，
     t 必须是容器或支撑面；place 内部自行完成下降、释放与撤离；
   - basket 是容器，只能作为 place 的 target，严禁 pick(basket)。
5. 典型顺序为 move_to(物体) -> pick(物体) -> move_to(目标) ->
   place(物体, 目标)；根据任务需要生成 2 到 6 个动作，不得有冗余动作。
6. 缺少必要物体或目标时，feasible 必须为 false，actions 必须为空数组。
7. 若存在自动验证反馈，必须先解决其中指出的违例动作，严禁原样重试。
8. 只输出 JSON，不要输出 JSON 之外的任何解释或 Markdown 代码块。

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
            environment_text=json.dumps(
                self.current_environment,
                ensure_ascii=False,
                indent=2
            ),
            object_text=object_text,
            skill_catalog=skill_catalog_text(),
            scene_text=scene_text,
            failure_text=failure_text,
            validation_text=validation_text
        )

    def _environment_object_names(self):
        names = []

        for obj in self.current_environment.get('objects', []):
            if isinstance(obj, dict):
                name = obj.get('name')
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())

        return names

    def _rejected_plan(self, command, reason, replans=0, planner='fallback'):
        return {
            'command': command,
            'schema_version': SCHEMA_VERSION,
            'feasible': False,
            'reason': reason,
            'failure_reason': reason,
            'actions': [],
            'planner': planner,
            'status': 'rejected',
            'replans': replans,
            'based_on_world_revision': (
                self.latest_scene_graph.world_revision
                if self.latest_scene_graph is not None else None
            ),
            'image_path': self.current_environment.get('image_path', ''),
        }

    @staticmethod
    def _check_plan_reachability(actions, graph):
        """Reject object references without positions in the workspace."""
        for action in actions:
            for key in ('object', 'target'):
                name = action.get(key)
                if not name:
                    continue
                node = graph.objects.get(name)
                if node is None or node.position is None:
                    return '{} has no metric position.'.format(name)
                x, y, z = node.position
                if not 0.25 <= x <= 0.85:
                    return '{} x={}m is outside [0.25, 0.85].'.format(
                        name, x
                    )
                if not -0.45 <= y <= 0.45:
                    return '{} y={}m is outside [-0.45, 0.45].'.format(
                        name, y
                    )
                if not 0.0 <= z <= 0.30:
                    return '{} z={}m is outside [0.0, 0.30].'.format(
                        name, z
                    )
        return ''

    def _mock_plan(self, command, graph):
        """Create a deterministic plan for offline ROS integration tests."""
        from agent_robot.scene_graph import GroundingError, resolve_instance

        normalized = command.strip().lower()
        object_reference = 'apple'
        if 'banana' in normalized or '香蕉' in normalized:
            object_reference = 'banana'
        elif 'cup' in normalized or '杯子' in normalized:
            object_reference = 'cup'
        colors = {
            'red': '红',
            'green': '绿',
            'blue': '蓝',
            'yellow': '黄',
        }
        for color, chinese in colors.items():
            if color in normalized or chinese in normalized:
                object_reference = '{} apple'.format(color)
                break
        target_reference = 'bin' if (
            'bin' in normalized
            or 'box' in normalized
            or '箱子' in normalized
            or '盒子' in normalized
        ) else 'basket'
        try:
            object_id = resolve_instance(object_reference, graph)
            target_id = resolve_instance(target_reference, graph)
        except GroundingError as error:
            return self._rejected_plan(
                command,
                'scene_grounding_failed: {}'.format(error),
                planner='mock',
            )

        actions = [
            {'skill': 'move_to', 'object': object_id},
            {'skill': 'pick', 'object': object_id},
            {'skill': 'move_to', 'object': target_id},
            {
                'skill': 'place',
                'object': object_id,
                'target': target_id,
            },
        ]
        evaluation = evaluate_generated_plan(
            {'feasible': True, 'actions': actions},
            self._grounding_names(graph),
            graph,
        )
        if not evaluation.accepted:
            return self._rejected_plan(
                command,
                'plan_validation_failed: {}'.format(
                    evaluation.error_detail
                ),
                planner='mock',
            )
        reachability_error = self._check_plan_reachability(
            evaluation.actions, graph
        )
        if reachability_error:
            return self._rejected_plan(
                command,
                'object_not_reachable: {}'.format(reachability_error),
                planner='mock',
            )
        return {
            'command': command,
            'schema_version': SCHEMA_VERSION,
            'feasible': True,
            'reason': '',
            'failure_reason': '',
            'actions': evaluation.actions,
            'planner': 'mock',
            'status': 'planned',
            'replans': 0,
            'based_on_world_revision': graph.world_revision,
            'image_path': self.current_environment.get('image_path', ''),
        }

    def fallback_plan(self, command, replans=0):
        return self._rejected_plan(
            command,
            '任务规划服务暂时不可用或输出不合法，无法安全生成计划。',
            replans=replans,
            planner='fallback'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlanner()
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
