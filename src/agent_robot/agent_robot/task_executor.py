"""Task executor: the single authority for structured skill execution.

Responsibilities (M2):

- consume schema-v2 plans (``actions`` list) from ``/task_plan``;
- ground action parameters against the latest ``/environment_state``;
- hold the authoritative :class:`~agent_robot.scene_graph.SceneGraph`
  and enforce skill preconditions both for the whole plan (before any
  motion) and per action (against the live graph);
- publish world-state updates on the latched ``/scene_graph`` topic;
- inject simulated failures from ``/simulate_failure``;
- publish the five execution states on ``/task_status``:
  ``action_started`` / ``action_completed`` / ``failed`` /
  ``succeeded`` / ``rejected``;
- dispatch skills to one backend (``mock`` or ``pybullet``).
"""

import hashlib
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from agent_robot.scene_graph import (
    GroundingError,
    SOURCE_EXECUTION,
    SceneGraph,
    ground_action,
    ground_name,
    validate_actions,
)
from agent_robot.skill_registry import (
    ActionSchemaError,
    SkillExecutionError,
    SkillRegistry,
    parse_actions,
)


# Latched so planners/subscribers started after the executor immediately
# receive the latest world state (M2).
SCENE_GRAPH_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


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

        self.graph_publisher = self.create_publisher(
            String,
            '/scene_graph',
            SCENE_GRAPH_QOS
        )

        self.current_environment = {
            'status': 'environment_not_received',
            'objects': []
        }

        # Authoritative symbolic world state. The fixed catalog scene is
        # valid for the mock backend and the fixed PyBullet scene until real
        # perception arrives; it must never masquerade as perception.
        self.scene_graph = SceneGraph.build_default()

        self.executing = False
        self.pending_failure = None
        self.last_plan_hash = None

        self.get_logger().info(
            'Task Executor started (backend={}).'.format(backend_name)
        )
        self.get_logger().info(
            'Scene graph initialized from catalog defaults (backend={}); '
            'waiting for perception.'.format(backend_name)
        )
        self._publish_scene_graph()

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
        self.scene_graph = SceneGraph.build_from_environment(environment)

        self.get_logger().info(
            'Environment state received with {} object(s).'.format(
                len(environment['objects'])
            )
        )

        for warning in self.scene_graph.warnings:
            self.get_logger().warning(
                'Scene graph builder warning: {}'.format(warning)
            )

        self._publish_scene_graph()

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

        # Whole-sequence validation against the current authoritative
        # graph: an infeasible sequence fails before any motion starts.
        report = validate_actions(grounded_actions, self.scene_graph)

        if not report.ok:
            violation = report.violation
            reason = 'precondition_violation: {}: {}'.format(
                violation.reason_code, violation.detail
            )
            self.get_logger().error(
                'Plan rejected before execution: {}'.format(reason)
            )
            self._publish_status(
                command=command,
                status='failed',
                action=violation.action,
                index=violation.index + 1,
                total=len(grounded_actions),
                reason=reason
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
            # Runtime precondition re-check against the live graph: never
            # start a motion whose preconditions are not satisfied.
            violation = self.scene_graph.check_preconditions(action)

            if violation is not None:
                reason_code, detail = violation
                reason = 'precondition_violation: {}: {}'.format(
                    reason_code, detail
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
                self._publish_scene_graph()
                return

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
                # No effect is applied on failure; publish the unchanged
                # graph so planners see the true failure-time state.
                self._publish_scene_graph()
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
                self._publish_scene_graph()
                return

            # Only a physically completed action advances the world state.
            self.scene_graph.apply(action)
            self.scene_graph.source = SOURCE_EXECUTION

            self._publish_status(
                command=command,
                status='action_completed',
                action=action,
                index=index,
                total=total
            )
            self._publish_scene_graph()

        self._publish_status(
            command=command,
            status='succeeded',
            total=total,
            reason='All {} action(s) completed.'.format(total)
        )
        self.get_logger().info('Task succeeded: {}'.format(command))

    def _publish_scene_graph(self):
        message = String()
        message.data = json.dumps(
            self.scene_graph.to_dict(),
            ensure_ascii=False
        )
        self.graph_publisher.publish(message)

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
