"""Task executor: the single authority for structured skill execution.

Responsibilities (M1):

- consume schema-v2 plans (``actions`` list) from ``/task_plan``;
- ground action parameters against the latest ``/environment_state``;
- validate actions through :mod:`agent_robot.skill_registry`;
- inject simulated failures from ``/simulate_failure``;
- publish the five execution states on ``/task_status``:
  ``action_started`` / ``action_completed`` / ``failed`` /
  ``succeeded`` / ``rejected``;
- dispatch skills to one backend (``mock`` or ``pybullet``).
"""

import hashlib
import json
import re
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from agent_robot.skill_registry import (
    ActionSchemaError,
    SkillAction,
    SkillExecutionError,
    SkillRegistry,
    parse_actions,
)


# Object names the simulated scene physically knows about.
CANONICAL_OBJECTS = ('apple', 'cup', 'basket')

_COLOR_WORDS = {
    'red', 'blue', 'brown', 'green', 'yellow',
    'black', 'white', 'orange', 'purple', 'gray', 'grey'
}

_CHINESE_ALIASES = {
    '苹果': 'apple',
    '篮子': 'basket',
    '篮筐': 'basket',
    '杯子': 'cup',
    '水杯': 'cup',
}


def semantic_tokens(text):
    # type: (str) -> List[str]
    """Extract color-free semantic tokens from an object name."""
    if text in _CHINESE_ALIASES:
        return [_CHINESE_ALIASES[text]]

    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [token for token in tokens if token not in _COLOR_WORDS]


def ground_name(query, env_names):
    # type: (str, List[str]) -> Optional[str]
    """Map an action parameter to a canonical scene object name.

    Matching order: normalized exact match, then semantic token overlap
    (color adjectives are ignored, so ``red_apple`` grounds ``apple``).
    Returns ``None`` when no environment object supports the query or the
    result is not a physically available object.
    """
    if not isinstance(query, str) or not query.strip():
        return None

    normalized_query = query.strip().lower()

    if normalized_query in _CHINESE_ALIASES:
        canonical = _CHINESE_ALIASES[normalized_query]
        if canonical in CANONICAL_OBJECTS:
            return canonical
        return None

    query_tokens = semantic_tokens(normalized_query)

    matched_env_name = None

    for env_name in env_names:
        normalized_env = env_name.strip().lower()

        if normalized_query == normalized_env:
            matched_env_name = normalized_env
            break

        env_tokens = semantic_tokens(normalized_env)

        if query_tokens and set(query_tokens) & set(env_tokens):
            matched_env_name = normalized_env
            break

    if matched_env_name is None:
        # The environment may be missing while the query itself is a
        # canonical scene name (debugging / mock scenarios).
        matched_env_name = normalized_query

    for token in semantic_tokens(matched_env_name):
        if token in CANONICAL_OBJECTS:
            return token

    return None


def ground_action(action, env_names):
    # type: (SkillAction, List[str]) -> SkillAction
    """Return a copy of *action* with grounded canonical object names."""
    grounded = {}

    for param_name, value in action.params.items():
        canonical = ground_name(value, env_names)

        if canonical is None:
            raise GroundingError(
                'Cannot ground {} parameter "{}" against scene objects: '
                '{}'.format(action.skill, param_name, value)
            )

        grounded[param_name] = canonical

    return SkillAction(skill=action.skill, params=grounded)


class GroundingError(ValueError):
    """An action parameter cannot be mapped to a scene object."""


class MockSkillBackend(object):
    """Deterministic backend without physics for headless runs."""

    def __init__(self, logger):
        self._logger = logger
        self._state = {'held_object': None}
        self._registry = SkillRegistry()
        self._registry.register('move_to', self._handle_move_to)
        self._registry.register('pick', self._handle_pick)
        self._registry.register('place', self._handle_place)

    def _handle_move_to(self, action, context):
        time.sleep(0.5)

    def _handle_pick(self, action, context):
        if self._state['held_object'] is not None:
            raise SkillExecutionError(
                'Gripper is already holding "{}".'.format(
                    self._state['held_object']
                )
            )

        time.sleep(0.5)
        self._state['held_object'] = action.object

    def _handle_place(self, action, context):
        if self._state['held_object'] != action.object:
            raise SkillExecutionError(
                'Cannot place "{}": the gripper is holding "{}".'.format(
                    action.object, self._state['held_object']
                )
            )

        time.sleep(0.5)
        self._state['held_object'] = None

    def execute(self, action):
        self._logger.info(
            '[mock] executing {}'.format(action.describe())
        )
        self._registry.execute(action)

    def shutdown(self):
        pass


class PyBulletSkillBackend(object):
    """PyBullet GUI backend; pybullet is imported lazily."""

    def __init__(self, logger):
        self._logger = logger

        import pybullet as p
        from agent_robot import pybullet_robot

        self._p = p
        self._pybullet_robot = pybullet_robot

        robot_id, object_table = pybullet_robot.create_scene(p.GUI)
        self._state = {'held_object': None, 'constraint_id': None}

        self._registry = SkillRegistry()

        for name, handler in pybullet_robot.build_skill_handlers(
            robot_id, object_table, self._state
        ).items():
            self._registry.register(name, handler)

        self._logger.info('PyBullet skill backend is ready.')

    def execute(self, action):
        self._registry.execute(action)

    def shutdown(self):
        if self._p.isConnected():
            self._p.disconnect()


class TaskExecutor(Node):

    def __init__(self):
        super().__init__('task_executor')

        self.declare_parameter('backend', 'mock')
        backend_name = self.get_parameter('backend').value

        if backend_name not in ('mock', 'pybullet'):
            self.get_logger().warning(
                'Unknown backend "{}", falling back to mock.'.format(
                    backend_name
                )
            )
            backend_name = 'mock'

        self._backend_name = backend_name
        self._backend = None

        self.plan_subscription = self.create_subscription(
            String,
            '/task_plan',
            self.plan_callback,
            10
        )

        self.environment_subscription = self.create_subscription(
            String,
            '/environment_state',
            self.environment_callback,
            10
        )

        self.failure_subscription = self.create_subscription(
            String,
            '/simulate_failure',
            self.failure_callback,
            10
        )

        self.status_publisher = self.create_publisher(
            String,
            '/task_status',
            10
        )

        self.current_environment = {
            'status': 'environment_not_received',
            'objects': []
        }
        self.executing = False
        self.pending_failure = None
        self.last_plan_hash = None

        self.get_logger().info(
            'Task Executor started (backend={}).'.format(backend_name)
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def environment_callback(self, msg):
        try:
            environment = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                'Invalid environment data: {}'.format(error)
            )
            return

        if not isinstance(environment.get('objects'), list):
            self.get_logger().error(
                'Environment is missing an objects list.'
            )
            return

        self.current_environment = environment
        self.get_logger().info(
            'Environment state received with {} object(s).'.format(
                len(environment['objects'])
            )
        )

    def failure_callback(self, msg):
        payload = msg.data.strip()

        if not payload:
            self.pending_failure = None
            return

        try:
            parsed = json.loads(payload)
            skill = parsed.get('skill')
            target_object = parsed.get('object')
        except (json.JSONDecodeError, AttributeError):
            skill = payload
            target_object = None

        if not isinstance(skill, str) or not skill.strip():
            self.get_logger().error(
                'Invalid failure injection payload: {}'.format(payload)
            )
            return

        self.pending_failure = {
            'skill': skill.strip(),
            'object': target_object
        }

        self.get_logger().warning(
            'Next action matching {} will be simulated as failed.'.format(
                self.pending_failure
            )
        )

    def plan_callback(self, msg):
        if self.executing:
            self.get_logger().warning(
                'Executor is busy; ignoring an incoming plan.'
            )
            return

        try:
            plan = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error('Invalid task plan JSON: {}'.format(error))
            return

        if 'actions' not in plan:
            self.get_logger().info(
                'Ignoring legacy plan without structured actions.'
            )
            return

        command = plan.get('command', '')
        feasible = plan.get('feasible', True)
        reason = plan.get('reason', '')
        raw_actions = plan.get('actions', [])

        plan_hash = self._hash_plan(command, feasible, raw_actions)

        if plan_hash == self.last_plan_hash:
            self.get_logger().info('Duplicate plan ignored.')
            return

        self.last_plan_hash = plan_hash

        self.get_logger().info('Received task: {}'.format(command))

        if not feasible:
            self.get_logger().warning('Task rejected: {}'.format(reason))
            self._publish_status(
                command=command,
                status='rejected',
                reason=reason
            )
            return

        try:
            actions = parse_actions(raw_actions)
        except ActionSchemaError as error:
            self.get_logger().error('Invalid action schema: {}'.format(error))
            self._publish_status(
                command=command,
                status='failed',
                reason='invalid_action_schema: {}'.format(error)
            )
            return

        env_names = [
            obj.get('name', '')
            for obj in self.current_environment.get('objects', [])
            if isinstance(obj, dict)
        ]

        grounded_actions = []

        for action in actions:
            try:
                grounded_actions.append(ground_action(action, env_names))
            except GroundingError as error:
                self.get_logger().error(str(error))
                self._publish_status(
                    command=command,
                    status='failed',
                    action=action,
                    index=len(grounded_actions) + 1,
                    total=len(actions),
                    reason='object_not_found: {}'.format(error)
                )
                return

        self.executing = True

        try:
            self._run_actions(command, grounded_actions)
        finally:
            self.executing = False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_actions(self, command, actions):
        backend = self._get_backend()

        if backend is None:
            self._publish_status(
                command=command,
                status='failed',
                reason='skill_backend_unavailable'
            )
            return

        total = len(actions)

        for index, action in enumerate(actions, start=1):
            if self._failure_matches(action):
                reason = (
                    '动作 {} 执行失败：注入的模拟故障。'.format(
                        action.describe()
                    )
                )
                self.get_logger().error(reason)
                self._publish_status(
                    command=command,
                    status='failed',
                    action=action,
                    index=index,
                    total=total,
                    reason=reason
                )
                self.pending_failure = None
                return

            self._publish_status(
                command=command,
                status='action_started',
                action=action,
                index=index,
                total=total
            )

            self.get_logger().info(
                'Executing action {}/{}: {}'.format(
                    index, total, action.describe()
                )
            )

            try:
                backend.execute(action)
            except SkillExecutionError as error:
                reason = '动作 {} 执行失败：{}'.format(
                    action.describe(), error
                )
                self.get_logger().error(reason)
                self._publish_status(
                    command=command,
                    status='failed',
                    action=action,
                    index=index,
                    total=total,
                    reason=reason
                )
                return

            self._publish_status(
                command=command,
                status='action_completed',
                action=action,
                index=index,
                total=total
            )

        self._publish_status(
            command=command,
            status='succeeded',
            total=total,
            reason='All {} action(s) completed.'.format(total)
        )
        self.get_logger().info('Task succeeded: {}'.format(command))

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            if self._backend_name == 'pybullet':
                self._backend = PyBulletSkillBackend(self.get_logger())
            else:
                self._backend = MockSkillBackend(self.get_logger())
        except Exception as error:
            self.get_logger().error(
                'Failed to initialize "{}" backend: {}'.format(
                    self._backend_name, error
                )
            )
            return None

        return self._backend

    def _failure_matches(self, action):
        if self.pending_failure is None:
            return False

        if self.pending_failure['skill'] != action.skill:
            return False

        wanted_object = self.pending_failure.get('object')

        if wanted_object:
            env_names = [
                obj.get('name', '')
                for obj in self.current_environment.get('objects', [])
                if isinstance(obj, dict)
            ]
            wanted_canonical = ground_name(wanted_object, env_names)
            return wanted_canonical == action.object

        return True

    @staticmethod
    def _hash_plan(command, feasible, raw_actions):
        serialized = json.dumps(
            {
                'command': command,
                'feasible': feasible,
                'actions': raw_actions
            },
            sort_keys=True,
            ensure_ascii=False
        )
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def _publish_status(
        self,
        command,
        status,
        action=None,
        index=None,
        total=None,
        reason=''
    ):
        data = {
            'command': command,
            'status': status,
            'action': action.to_dict() if action is not None else None,
            'step_index': index,
            'total_steps': total,
            'reason': reason
        }

        if action is not None:
            # On failure mark the readable step explicitly so existing
            # status-string consumers (2D UI) still detect the failure.
            if status == 'failed':
                data['step'] = '{} failed'.format(action.describe())
            else:
                data['step'] = action.describe()
        else:
            data['step'] = ''

        message = String()
        message.data = json.dumps(data, ensure_ascii=False)
        self.status_publisher.publish(message)

    def destroy_node(self):
        if self._backend is not None:
            try:
                self._backend.shutdown()
            except Exception as error:
                self.get_logger().warning(
                    'Backend shutdown error: {}'.format(error)
                )

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TaskExecutor()

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
