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
import os
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
from agent_robot.perception.coordinate_mapper import (
    correct_simulation_placements,
)
from agent_robot.ros_qos import ENVIRONMENT_STATE_QOS
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

    def __init__(self, logger, action_duration=0.05):
        self._logger = logger
        self._action_duration = float(action_duration)
        self._state = {'held_object': None}
        self._registry = SkillRegistry()
        self._registry.register('move_to', self._handle_move_to)
        self._registry.register('pick', self._handle_pick)
        self._registry.register('place', self._handle_place)

    def _handle_move_to(self, action, context):
        time.sleep(self._action_duration)

    def _handle_pick(self, action, context):
        if self._state['held_object'] is not None:
            raise SkillExecutionError(
                'Gripper is already holding "{}".'.format(
                    self._state['held_object']
                )
            )

        time.sleep(self._action_duration)
        self._state['held_object'] = action.object

    def _handle_place(self, action, context):
        if self._state['held_object'] != action.object:
            raise SkillExecutionError(
                'Cannot place "{}": the gripper is holding "{}".'.format(
                    action.object, self._state['held_object']
                )
            )

        time.sleep(self._action_duration)
        self._state['held_object'] = None

    def execute(self, action):
        self._logger.info(
            '[mock] executing {}'.format(action.describe())
        )
        self._registry.execute(action)

    def shutdown(self):
        pass


class PyBulletSkillBackend(object):
    """PyBullet backend; PyBullet is imported lazily."""

    def __init__(
        self,
        logger,
        node,
        spec=None,
        connection_mode=None,
    ):
        self._logger = logger
        self._node = node

        import pybullet as p
        from agent_robot import pybullet_robot

        self._p = p
        self._pybullet_robot = pybullet_robot
        if connection_mode is None:
            connection_mode = p.GUI
        robot_id, object_table = pybullet_robot.create_scene(
            spec=spec,
            connection_mode=connection_mode,
        )
        self.object_table = object_table
        self._state = {'held_object': None, 'constraint_id': None}
        self._trajectory_cursor = 0

        self._registry = SkillRegistry()

        for name, handler in pybullet_robot.build_skill_handlers(
            robot_id, object_table, self._state
        ).items():
            self._registry.register(name, handler)

        self._logger.info('PyBullet skill backend is ready.')

    def apply_world_event(self, event):
        return self._pybullet_robot.apply_world_event(
            self.object_table,
            event,
        )

    def execute(self, action):
        self._registry.execute(action)

    def consume_trajectory_points(self):
        points = self._pybullet_robot._MOTION_TRACE[
            self._trajectory_cursor:
        ]
        self._trajectory_cursor = len(self._pybullet_robot._MOTION_TRACE)
        return [dict(point) for point in points]

    def shutdown(self):
        if self._p.isConnected():
            self._p.disconnect()


class TaskExecutor(Node):

    def __init__(self):
        super().__init__('task_executor')

        self.declare_parameter('backend', 'mock')
        self.declare_parameter('mock_action_duration', 0.05)
        self.declare_parameter('scenario_spec', '')
        self.declare_parameter('category', '')
        self.declare_parameter('seed', -1)
        self.declare_parameter('pybullet_connection_mode', 'GUI')
        self.declare_parameter('require_perception_scene', False)
        backend_name = self.get_parameter('backend').value
        self._mock_action_duration = float(
            self.get_parameter('mock_action_duration').value
        )
        self._scenario_spec_path = self.get_parameter(
            'scenario_spec'
        ).value
        self._scenario_category = self.get_parameter('category').value
        seed_value = self.get_parameter('seed').value
        self._scenario_seed = (
            int(seed_value) if int(seed_value) >= 0 else None
        )
        self._pybullet_connection_mode = self.get_parameter(
            'pybullet_connection_mode'
        ).value
        self._require_perception_scene = bool(
            self.get_parameter('require_perception_scene').value
        )

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
            ENVIRONMENT_STATE_QOS
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
        self.trajectory_publisher = self.create_publisher(
            String,
            '/trajectory_point',
            10,
        )

        self.graph_publisher = self.create_publisher(
            String,
            '/scene_graph',
            SCENE_GRAPH_QOS
        )
        self.world_event_ack_publisher = self.create_publisher(
            String,
            '/world_event_ack',
            10,
        )
        self.world_event_subscription = self.create_subscription(
            String,
            '/world_event',
            self.world_event_callback,
            10,
        )

        self.current_environment = {
            'status': 'environment_not_received',
            'objects': []
        }
        self._perception_scene_id = None
        self._cached_scenario_id = None

        # Authoritative symbolic world state. The fixed catalog scene is
        # valid for the mock backend and the fixed PyBullet scene until real
        # perception arrives; it must never masquerade as perception.
        self.scene_graph = SceneGraph.build_default()

        self.executing = False
        self.pending_failure = None
        self.last_plan_hash = None

        # M5 action-level state machine (revision-aware episodes).
        self._revision_protocol = False
        self._belief_initialized = False
        self._cached_environment = None
        self._cached_revision = -1
        self._active_command = ''
        self._active_actions = []
        self._active_index = 0
        self._active_plan_revision = None
        self._motion_busy = False
        self._step_timer = self.create_timer(0.04, self._execution_step)

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

        revision = environment.get('world_revision')
        scenario_id = environment.get('scenario_id')
        if environment.get('source') == 'perception':
            if (
                isinstance(scenario_id, str)
                and scenario_id != self._cached_scenario_id
            ):
                if self.executing:
                    self.get_logger().error(
                        'Cannot replace the perceived scene while executing.'
                    )
                    return
                if self._backend is not None:
                    self._backend.shutdown()
                    self._backend = None
                self._cached_revision = -1
                self.last_plan_hash = None
                self._cached_scenario_id = scenario_id
                self._perception_scene_id = scenario_id
            self.current_environment = environment
            self._belief_initialized = True
            self.scene_graph.initialize_from_environment(environment)
            self._publish_scene_graph()
            self._cached_environment = environment
            self._cached_revision = (
                revision if isinstance(revision, int) else 0
            )
            self._revision_protocol = isinstance(revision, int)
            return

        if isinstance(revision, int):
            self._revision_protocol = True
            if revision > self._cached_revision:
                self._cached_environment = environment
                self._cached_revision = revision
                self.current_environment = environment
            return

        self.current_environment = environment
        self.scene_graph = SceneGraph.build_from_environment(environment)
        self._belief_initialized = True

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
            failure_code = parsed.get('failure_code')
            skill = parsed.get('skill')
            target_object = parsed.get('object')
        except (json.JSONDecodeError, AttributeError):
            failure_code = (
                'grasp_failed' if payload == 'grasp_failed' else None
            )
            skill = 'pick' if failure_code else payload
            target_object = None

        if failure_code == 'grasp_failed':
            skill = 'pick'

        if not isinstance(skill, str) or not skill.strip():
            self.get_logger().error(
                'Invalid failure injection payload: {}'.format(payload)
            )
            return

        self.pending_failure = {
            'skill': skill.strip(),
            'object': target_object,
            'failure_code': (
                failure_code
                or ('grasp_failed' if skill.strip() == 'pick' else None)
            ),
        }

        self.get_logger().warning(
            'Next action matching {} will be simulated as failed.'.format(
                self.pending_failure
            )
        )

    def world_event_callback(self, msg):
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                'Invalid world event JSON: {}'.format(error)
            )
            return

        if self._backend_name != 'pybullet':
            self.get_logger().warning(
                'Ignoring physical world event with non-PyBullet backend.'
            )
            return

        backend = self._get_backend()
        if backend is None:
            acknowledgement = {
                'revision': event.get('revision'),
                'applied': False,
                'reason': 'pybullet_backend_unavailable',
            }
        else:
            try:
                acknowledgement = backend.apply_world_event(event)
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                self.get_logger().error(
                    'Failed to apply world event: {}'.format(error)
                )
                acknowledgement = {
                    'revision': event.get('revision'),
                    'applied': False,
                    'reason': str(error),
                }

        message = String()
        message.data = json.dumps(acknowledgement, ensure_ascii=False)
        self.world_event_ack_publisher.publish(message)

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
        based_on_revision = plan.get('based_on_world_revision')
        if based_on_revision is not None:
            based_on_revision = int(based_on_revision)

        plan_hash = self._hash_plan(
            command,
            feasible,
            raw_actions,
            based_on_revision,
            plan.get('replans', 0),
        )

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

        if self._revision_protocol or based_on_revision is not None:
            self._drain_environment(force_initialize=True)

        grounding_context = (
            self.scene_graph
            if self._revision_protocol or self._belief_initialized
            else [
                obj.get('name', obj.get('id', ''))
                for obj in self.current_environment.get('objects', [])
                if isinstance(obj, dict)
            ]
        )

        grounded_actions = []

        for action in actions:
            try:
                grounded_actions.append(
                    ground_action(action, grounding_context)
                )
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

        if self._revision_protocol or based_on_revision is not None:
            self._start_revision_plan(
                command,
                grounded_actions,
                based_on_revision,
            )
            return

        self.executing = True

        try:
            self._run_actions(command, grounded_actions)
        finally:
            self.executing = False

    def _start_revision_plan(self, command, actions, based_on_revision):
        if self.executing:
            self.get_logger().warning(
                'Executor is busy; ignoring revision-aware plan.'
            )
            return

        self._drain_environment(force_initialize=True)
        self._active_command = command
        self._active_actions = actions
        self._active_index = 0
        self._active_plan_revision = based_on_revision
        self.executing = True

    def _execution_step(self):
        if self._motion_busy:
            return
        if not self.executing or not self._active_actions:
            return

        self._drain_environment()

        if self._active_index >= len(self._active_actions):
            self._finish_revision_plan(success=True)
            return

        if not self._passes_revision_fence():
            return

        action = self._active_actions[self._active_index]
        index = self._active_index + 1
        total = len(self._active_actions)
        command = self._active_command

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
                reason=reason,
            )
            self._publish_scene_graph()
            self._finish_revision_plan(success=False)
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
                reason=reason,
                failure_code=(
                    'grasp_failed' if action.skill == 'pick' else None
                ),
            )
            self.pending_failure = None
            self._publish_scene_graph()
            self._finish_revision_plan(success=False)
            return

        self._publish_status(
            command=command,
            status='action_started',
            action=action,
            index=index,
            total=total,
        )

        backend = self._get_backend()
        if backend is None:
            self._publish_status(
                command=command,
                status='failed',
                reason='skill_backend_unavailable',
            )
            self._finish_revision_plan(success=False)
            return

        self._motion_busy = True
        try:
            backend.execute(action)
        except SkillExecutionError as error:
            self._publish_backend_trajectory(
                backend, command, index, action
            )
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
                reason=reason,
                failure_code=(
                    'grasp_failed' if action.skill == 'pick' else None
                ),
            )
            self._publish_scene_graph()
            self._finish_revision_plan(success=False)
            return
        finally:
            self._motion_busy = False

        self._publish_backend_trajectory(
            backend, command, index, action
        )

        self.scene_graph.apply_action_effect(action)
        self.scene_graph.source = SOURCE_EXECUTION

        self._publish_status(
            command=command,
            status='action_completed',
            action=action,
            index=index,
            total=total,
        )
        self._publish_scene_graph()
        self._active_index += 1

        if self._active_index >= len(self._active_actions):
            self._finish_revision_plan(success=True)

    def _finish_revision_plan(self, success):
        if success:
            self._publish_status(
                command=self._active_command,
                status='succeeded',
                total=len(self._active_actions),
                reason='All {} action(s) completed.'.format(
                    len(self._active_actions)
                ),
            )
            self.get_logger().info(
                'Task succeeded: {}'.format(self._active_command)
            )

        self._active_actions = []
        self._active_command = ''
        self._active_index = 0
        self._active_plan_revision = None
        self.executing = False

    def _passes_revision_fence(self):
        current = self.scene_graph.world_revision
        plan_revision = self._active_plan_revision

        if plan_revision is None or current <= plan_revision:
            return True

        remaining = self._active_actions[self._active_index:]
        report = validate_actions(remaining, self.scene_graph)

        if report.ok:
            self._active_plan_revision = current
            return True

        violation = report.violation
        reason = 'stale_plan: {}: {}'.format(
            violation.reason_code, violation.detail
        )
        self.get_logger().error(
            'Plan rejected at revision fence: {}'.format(reason)
        )
        self._publish_status(
            command=self._active_command,
            status='failed',
            action=violation.action,
            index=violation.index + self._active_index + 1,
            total=len(self._active_actions),
            reason=reason,
        )
        self._publish_scene_graph()
        self._finish_revision_plan(success=False)
        return False

    def _drain_environment(self, force_initialize=False):
        environment = self._cached_environment

        if environment is None:
            return

        if not self._revision_protocol and not force_initialize:
            return

        if not self._belief_initialized or force_initialize:
            self.scene_graph.initialize_from_environment(environment)
            self._belief_initialized = True
        else:
            self.scene_graph.update_observation(environment)

        for warning in self.scene_graph.warnings:
            self.get_logger().warning(
                'Scene graph merge warning: {}'.format(warning)
            )

        self._publish_scene_graph()

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
                    reason=reason,
                    failure_code=(
                        self.pending_failure.get('failure_code')
                    ),
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
                    reason=reason,
                    failure_code=(
                        'grasp_failed' if action.skill == 'pick' else None
                    ),
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
                self._publish_backend_trajectory(
                    backend, command, index, action
                )
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

            self._publish_backend_trajectory(
                backend, command, index, action
            )

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
                import pybullet as p

                spec = self._load_scenario_spec()
                if spec is None:
                    spec = self._load_perception_spec()
                if self._require_perception_scene and spec is None:
                    raise ValueError(
                        'Waiting for an image-derived scene before '
                        'initializing PyBullet.'
                    )
                connection_mode = {
                    'GUI': p.GUI,
                    'DIRECT': p.DIRECT,
                }.get(self._pybullet_connection_mode.upper())
                if connection_mode is None:
                    raise ValueError(
                        'pybullet_connection_mode must be GUI or DIRECT.'
                    )
                self._backend = PyBulletSkillBackend(
                    self.get_logger(),
                    self,
                    spec=spec,
                    connection_mode=connection_mode,
                )
            else:
                self._backend = MockSkillBackend(
                    self.get_logger(),
                    action_duration=self._mock_action_duration,
                )
        except Exception as error:
            self.get_logger().error(
                'Failed to initialize "{}" backend: {}'.format(
                    self._backend_name, error
                )
            )
            return None

        return self._backend

    def _load_scenario_spec(self):
        if self._scenario_spec_path:
            from agent_robot.scenarios.spec import scene_spec_from_dict

            path = os.path.abspath(os.path.expanduser(
                self._scenario_spec_path
            ))
            with open(path, encoding='utf-8') as handle:
                return scene_spec_from_dict(json.load(handle))
        if self._scenario_category and self._scenario_seed is not None:
            from agent_robot.scenarios.generator import generate_scenario

            return generate_scenario(
                self._scenario_category,
                self._scenario_seed,
            )
        return None

    def _load_perception_spec(self):
        environment = self.current_environment
        if environment.get('source') != 'perception':
            return None

        from agent_robot.scenarios.spec import ObjectInstance

        raw_objects = environment.get('simulation_objects')
        placement_report = environment.get('simulation_placement')
        placement_is_precomputed = (
            isinstance(raw_objects, list)
            and isinstance(placement_report, dict)
        )
        if placement_is_precomputed:
            placement = {
                'objects': raw_objects,
                'warnings': placement_report.get('warnings', []),
                'adjustments': [],
                'containment': placement_report.get('containment', []),
            }
        else:
            raw_objects = raw_objects or environment.get('objects', [])
            placement = correct_simulation_placements(raw_objects)
        for warning in placement['warnings']:
            self.get_logger().warning(warning)
        for adjustment in placement['adjustments']:
            self.get_logger().warning(
                'Using corrected simulation-only position for {}: {} -> {}'
                .format(
                    adjustment['object'],
                    adjustment['perception_position'],
                    adjustment['simulation_position'],
                )
            )
        instances = []
        for raw in placement['objects']:
            object_type = raw.get('type')
            if object_type not in ('apple', 'basket', 'cup', 'block', 'bin'):
                continue
            position = raw.get('position')
            if (
                not isinstance(position, (list, tuple))
                or len(position) != 3
            ):
                raise ValueError(
                    'Perceived object {} has no valid position.'.format(
                        raw.get('id', raw.get('name'))
                    )
                )
            instances.append(ObjectInstance(
                instance_id=raw.get('id', object_type),
                object_type=object_type,
                color=raw.get('color'),
                position=tuple(float(value) for value in position),
                support=raw.get('support', 'table'),
                visible=bool(raw.get('visible', True)),
            ))
        if not instances:
            raise ValueError(
                'Perception environment has no supported object instances.'
            )

        class PerceivedScene(object):
            def __init__(
                self,
                objects,
                scenario_id,
                allowed_containment,
                simulation_placement_applied,
            ):
                self.source = 'perception'
                self.objects = tuple(objects)
                self.scenario_id = scenario_id
                self.simulation_placement_applied = (
                    simulation_placement_applied
                )
                self.allowed_containment = tuple(
                    tuple(pair) for pair in allowed_containment
                )

        return PerceivedScene(
            instances,
            self._perception_scene_id or 'perceived_scene',
            placement['containment'],
            placement_is_precomputed,
        )

    def _publish_backend_trajectory(
        self, backend, command, index, action
    ):
        consume_points = getattr(
            backend, 'consume_trajectory_points', None
        )
        if not callable(consume_points):
            return
        points = consume_points()
        if not points:
            self.get_logger().warning(
                'No motion samples captured for action {}.'.format(index)
            )
            return
        for point in points:
            message = String()
            message.data = json.dumps({
                'command': command,
                'action_index': index,
                'action': action.to_dict(),
                'point': point,
            })
            self.trajectory_publisher.publish(message)
        self.get_logger().info(
            'Published {} motion sample(s) for action {}.'.format(
                len(points), index
            )
        )

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
    def _hash_plan(
        command,
        feasible,
        raw_actions,
        based_on_world_revision=None,
        plan_attempt=0,
    ):
        payload = {
            'command': command,
            'feasible': feasible,
            'actions': raw_actions,
        }
        if based_on_world_revision is not None:
            payload['based_on_world_revision'] = based_on_world_revision
        payload['plan_attempt'] = plan_attempt

        serialized = json.dumps(
            payload,
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
        reason='',
        failure_code=None,
    ):
        data = {
            'command': command,
            'status': status,
            'action': action.to_dict() if action is not None else None,
            'step_index': index,
            'total_steps': total,
            'reason': reason,
            'world_revision': self.scene_graph.world_revision,
        }
        if failure_code:
            data['failure_code'] = failure_code

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
