"""ROS 2 world truth publisher for M5 parameterized episodes."""

import json
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from agent_robot.ros_qos import ENVIRONMENT_STATE_QOS
from agent_robot.scenarios.generator import generate_scenario
from agent_robot.scenarios.spec import scene_spec_from_dict
from agent_robot.scenarios.world_model import (
    EventPreconditionError,
    WorldModel,
)


def load_scene_spec_from_params(
    scenario_spec_path,
    category,
    seed,
    scenario_id,
):
    """
    Resolve a :class:`SceneSpec` from launch/node parameters.

    Either ``scenario_spec_path`` or ``category``+``seed`` must be set.
    """
    if scenario_spec_path:
        if category or seed is not None:
            raise ValueError(
                'Use either scenario_spec or category+seed, not both.'
            )
        path = os.path.abspath(os.path.expanduser(scenario_spec_path))
        with open(path, 'r', encoding='utf-8') as handle:
            return scene_spec_from_dict(json.load(handle))

    if not category or seed is None:
        raise ValueError(
            'Provide scenario_spec or both category and seed.'
        )

    return generate_scenario(category, int(seed), scenario_id=scenario_id)


class WorldNode(Node):

    ACK_TIMEOUT_SEC = 5.0

    def __init__(self):
        super().__init__('world_node')

        self.declare_parameter('scenario_spec', '')
        self.declare_parameter('category', '')
        self.declare_parameter('seed', -1)
        self.declare_parameter('scenario_id', '')
        self.declare_parameter('backend', 'mock')
        self.declare_parameter('ack_timeout_sec', self.ACK_TIMEOUT_SEC)

        spec_path = self.get_parameter('scenario_spec').value
        category = self.get_parameter('category').value
        seed_param = self.get_parameter('seed').value
        scenario_id = self.get_parameter('scenario_id').value or None
        backend = self.get_parameter('backend').value
        ack_timeout = float(self.get_parameter('ack_timeout_sec').value)

        seed = None if seed_param == -1 else int(seed_param)

        try:
            spec = load_scene_spec_from_params(
                spec_path if spec_path else None,
                category if category else None,
                seed,
                scenario_id,
            )
        except (ValueError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                'Failed to load scenario: {}'.format(error)
            )

        physical = backend == 'pybullet'
        self.world = WorldModel(spec, physical=physical)
        self._ack_timeout = ack_timeout
        self._pending_ack_revision = None
        self._pending_ack_deadline = None

        self.environment_publisher = self.create_publisher(
            String,
            '/environment_state',
            ENVIRONMENT_STATE_QOS,
        )
        self.status_subscription = self.create_subscription(
            String,
            '/task_status',
            self.status_callback,
            10,
        )
        self.ack_subscription = self.create_subscription(
            String,
            '/world_event_ack',
            self.ack_callback,
            10,
        )
        self.world_event_publisher = self.create_publisher(
            String,
            '/world_event',
            10,
        )
        self.failure_publisher = self.create_publisher(
            String,
            '/simulate_failure',
            10,
        )

        self.create_timer(0.05, self._tick)
        self.create_timer(0.2, self._publish_initial_once)
        self._initial_sent = False

        self.get_logger().info(
            'World node ready: scenario_id={} backend={}'.format(
                spec.scenario_id, backend
            )
        )

    def _publish_initial_once(self):
        if self._initial_sent:
            return
        self._initial_sent = True
        self._publish_environment()

    def _tick(self):
        if self._pending_ack_deadline is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now < self._pending_ack_deadline:
            return
        self.get_logger().error(
            'world_event ack timed out for revision {}.'.format(
                self._pending_ack_revision
            )
        )
        self.world.on_event_ack(self._pending_ack_revision, False)
        self._clear_ack_wait()
        self._publish_environment()

    def status_callback(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                'Invalid task status JSON: {}'.format(error)
            )
            return

        state = status.get('status')
        action = status.get('action') or {}
        skill = action.get('skill')
        params = action.get('params') or {}
        index = status.get('step_index')

        if not isinstance(skill, str) or not isinstance(index, int):
            return

        object_name = params.get('object')
        target_name = params.get('target')

        if state == 'action_started':
            inject = self.world.on_action_started(
                skill, object_name, index
            )
            if inject is not None:
                payload = {'skill': inject.skill}
                if inject.object:
                    payload['object'] = inject.object
                message = String()
                message.data = json.dumps(payload, ensure_ascii=False)
                self.failure_publisher.publish(message)
            return

        if state != 'action_completed':
            return

        try:
            result = self.world.on_action_completed(
                skill,
                object_name,
                target_name,
                index,
            )
        except EventPreconditionError as error:
            self.get_logger().error(
                'World transition rejected: {}'.format(error)
            )
            return

        if result.error:
            self.get_logger().warning(
                'World transition skipped: {}'.format(result.error)
            )
            return

        for command in self.world.drain_commands():
            event_message = String()
            event_message.data = json.dumps(
                {
                    'revision': command.revision,
                    'event': command.event,
                    'target': command.target,
                    'new_position': list(command.new_position),
                    'new_support': command.new_support,
                    'new_visibility': command.new_visibility,
                },
                ensure_ascii=False,
            )
            self.world_event_publisher.publish(event_message)
            if self.world.physical:
                self._pending_ack_revision = command.revision
                now = self.get_clock().now().nanoseconds / 1e9
                self._pending_ack_deadline = now + self._ack_timeout
            else:
                self._publish_environment()

        if not self.world.physical and result.environment_changed:
            self._publish_environment()

    def ack_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                'Invalid world_event_ack JSON: {}'.format(error)
            )
            return

        revision = payload.get('revision')
        applied = payload.get('applied', payload.get('ok', False))
        position = payload.get('actual_position')

        if not isinstance(revision, int):
            return

        actual = None
        if isinstance(position, (list, tuple)) and len(position) == 3:
            actual = (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )

        if self.world.on_event_ack(revision, bool(applied), actual):
            self._clear_ack_wait()
            self._publish_environment()

    def _clear_ack_wait(self):
        self._pending_ack_revision = None
        self._pending_ack_deadline = None

    def _publish_environment(self):
        payload = self.world.environment_payload()
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.environment_publisher.publish(message)
        self.get_logger().info(
            'Published environment revision {} ({} objects).'.format(
                payload.get('world_revision'),
                len(payload.get('objects', [])),
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = WorldNode()
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
