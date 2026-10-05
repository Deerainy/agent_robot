"""PyBullet scene and skill primitives for EmbodiedPlan.

Layer layout (M1):

- Internal primitives (not exposed to the LLM):
  ``move_to_position``, ``open_gripper``, ``close_gripper``.
- High-level skills registered into :class:`SkillRegistry`:
  ``move_to``, ``pick``, ``place``.

``create_scene`` supports both ``p.GUI`` (live demo) and ``p.DIRECT``
(headless verification). In DIRECT mode motion does not sleep in wall-clock
time, while the number of simulated physics steps is unchanged.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import pybullet as p
import pybullet_data

from agent_robot.skill_registry import (
    SkillExecutionError,
    SkillRegistry,
    parse_actions,
)


# When False (p.DIRECT), motion loops only step the physics without sleeping.
_REALTIME = True

_END_EFFECTOR_LINK = 11
_FINGER_JOINTS = [9, 10]
_PHYSICS_FREQUENCY = 240


@dataclass
class ObjectSpec:
    """Physical body and skill waypoints of one scene object."""

    name: str
    body_id: int
    position: List[float]
    above_position: List[float]
    grasp_position: Optional[List[float]] = None
    place_release_position: Optional[List[float]] = None
    place_rest_position: Optional[List[float]] = None
    graspable: bool = True


def create_box(position, half_extents, color):
    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )


def _sleep_steps(simulation_steps):
    """Step physics simulation_steps times, sleeping only in GUI mode."""
    for _ in range(simulation_steps):
        p.stepSimulation()

        if _REALTIME:
            time.sleep(1.0 / _PHYSICS_FREQUENCY)


def create_scene(connection_mode=p.GUI):
    """Build the tabletop scene.

    Returns ``(robot_id, object_table)`` where *object_table* maps canonical
    object names (``apple`` / ``cup`` / ``basket``) to :class:`ObjectSpec`.
    """
    global _REALTIME
    _REALTIME = (connection_mode == p.GUI)

    physics_client = p.connect(connection_mode)

    if physics_client < 0:
        raise RuntimeError(
            'Failed to connect to PyBullet (mode {}).'.format(connection_mode)
        )

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    if connection_mode == p.GUI:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.6,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.45, 0.0, 0.25]
        )

    # Ground and table.
    p.loadURDF('plane.urdf')
    p.loadURDF(
        'table/table.urdf',
        basePosition=[0.5, 0.0, -0.65],
        useFixedBase=True
    )

    robot_id = p.loadURDF(
        'franka_panda/panda.urdf',
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True
    )

    initial_joint_positions = [
        0.0,
        -0.45,
        0.0,
        -2.2,
        0.0,
        1.8,
        0.8
    ]

    for joint_index, joint_position in enumerate(initial_joint_positions):
        p.resetJointState(robot_id, joint_index, joint_position)

    object_table = {}

    # Red apple.
    apple_position = [0.55, -0.20, 0.06]

    apple_collision = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=0.045
    )
    apple_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.045,
        rgbaColor=[0.9, 0.05, 0.05, 1.0]
    )
    apple_id = p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=apple_collision,
        baseVisualShapeIndex=apple_visual,
        basePosition=apple_position
    )
    p.changeDynamics(
        apple_id,
        -1,
        lateralFriction=0.9,
        rollingFriction=0.05,
        spinningFriction=0.05,
        restitution=0.0
    )

    object_table['apple'] = ObjectSpec(
        name='apple',
        body_id=apple_id,
        position=list(apple_position),
        above_position=[0.55, -0.20, 0.32],
        grasp_position=[0.55, -0.20, 0.085]
    )

    # Blue cup.
    cup_position = [0.55, 0.05, 0.06]

    cup_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=0.045,
        height=0.10
    )
    cup_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.045,
        length=0.10,
        rgbaColor=[0.05, 0.35, 0.95, 1.0]
    )
    cup_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cup_collision,
        baseVisualShapeIndex=cup_visual,
        basePosition=cup_position
    )
    p.changeDynamics(
        cup_id,
        -1,
        lateralFriction=0.9,
        rollingFriction=0.05,
        spinningFriction=0.05,
        restitution=0.0
    )

    object_table['cup'] = ObjectSpec(
        name='cup',
        body_id=cup_id,
        position=list(cup_position),
        above_position=[0.55, 0.05, 0.32],
        grasp_position=[0.55, 0.05, 0.085]
    )

    # Brown basket (base + four walls).
    basket_color = [0.45, 0.22, 0.08, 1.0]

    create_box([0.70, 0.28, 0.025], [0.13, 0.10, 0.025], basket_color)
    create_box([0.70, 0.18, 0.09], [0.13, 0.015, 0.09], basket_color)
    create_box([0.70, 0.38, 0.09], [0.13, 0.015, 0.09], basket_color)
    create_box([0.57, 0.28, 0.09], [0.015, 0.10, 0.09], basket_color)
    create_box([0.83, 0.28, 0.09], [0.015, 0.10, 0.09], basket_color)

    object_table['basket'] = ObjectSpec(
        name='basket',
        body_id=-1,
        position=[0.70, 0.28, 0.025],
        above_position=[0.70, 0.28, 0.38],
        grasp_position=None,
        place_release_position=[0.70, 0.28, 0.14],
        place_rest_position=[0.70, 0.28, 0.105],
        graspable=False
    )

    return robot_id, object_table


# ---------------------------------------------------------------------------
# Internal motion primitives (never registered as plan skills)
# ---------------------------------------------------------------------------

def move_to_position(robot_id, target_position, duration=2.0):
    """Move the end effector to target_position via inverse kinematics."""
    downward_orientation = p.getQuaternionFromEuler(
        [math.pi, 0.0, 0.0]
    )

    joint_targets = p.calculateInverseKinematics(
        robot_id,
        _END_EFFECTOR_LINK,
        target_position,
        downward_orientation,
        maxNumIterations=200,
        residualThreshold=1e-5
    )

    simulation_steps = int(duration * _PHYSICS_FREQUENCY)

    for _ in range(simulation_steps):
        for joint_index in range(7):
            p.setJointMotorControl2(
                robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joint_targets[joint_index],
                force=200
            )

        p.stepSimulation()

        if _REALTIME:
            time.sleep(1.0 / _PHYSICS_FREQUENCY)


def open_gripper(robot_id, duration=0.7):
    """Open the Panda gripper fingers."""
    _control_gripper(robot_id, 0.04, duration)


def close_gripper(robot_id, duration=0.7):
    """Close the Panda gripper fingers."""
    _control_gripper(robot_id, 0.0, duration)


def _control_gripper(robot_id, opening, duration):
    simulation_steps = int(duration * _PHYSICS_FREQUENCY)

    for _ in range(simulation_steps):
        for joint_index in _FINGER_JOINTS:
            p.setJointMotorControl2(
                robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=opening,
                force=100
            )

        p.stepSimulation()

        if _REALTIME:
            time.sleep(1.0 / _PHYSICS_FREQUENCY)


def attach_object(robot_id, body_id):
    """Fix a grasped body to the end effector and disable its collisions."""
    link_state = p.getLinkState(
        robot_id,
        _END_EFFECTOR_LINK,
        computeForwardKinematics=True
    )

    body_position, body_orientation = (
        p.getBasePositionAndOrientation(body_id)
    )

    inverse_position, inverse_orientation = p.invertTransform(
        link_state[4],
        link_state[5]
    )

    relative_position, relative_orientation = p.multiplyTransforms(
        inverse_position,
        inverse_orientation,
        body_position,
        body_orientation
    )

    for link_index in range(-1, p.getNumJoints(robot_id)):
        p.setCollisionFilterPair(
            robot_id,
            body_id,
            link_index,
            -1,
            enableCollision=0
        )

    return p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=_END_EFFECTOR_LINK,
        childBodyUniqueId=body_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=relative_position,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=relative_orientation,
        childFrameOrientation=[0, 0, 0, 1]
    )


def _restore_collision(robot_id, body_id):
    for link_index in range(-1, p.getNumJoints(robot_id)):
        p.setCollisionFilterPair(
            robot_id,
            body_id,
            link_index,
            -1,
            enableCollision=1
        )


# ---------------------------------------------------------------------------
# High-level skills (the implementations behind SKILL_SPECS)
# ---------------------------------------------------------------------------

def pick_object(robot_id, spec, state):
    """Pick a graspable object and update the execution state dict."""
    if not isinstance(spec, ObjectSpec):
        raise SkillExecutionError(
            'pick requires an ObjectSpec target.'
        )

    if not spec.graspable or spec.grasp_position is None:
        raise SkillExecutionError(
            'Object "{}" cannot be grasped.'.format(spec.name)
        )

    if state.get('held_object') is not None:
        raise SkillExecutionError(
            'Gripper is already holding "{}".'.format(
                state['held_object']
            )
        )

    open_gripper(robot_id)
    move_to_position(robot_id, spec.above_position, duration=2.0)
    move_to_position(robot_id, spec.grasp_position, duration=1.5)
    close_gripper(robot_id)

    constraint_id = attach_object(robot_id, spec.body_id)
    move_to_position(robot_id, spec.above_position, duration=1.5)

    state['held_object'] = spec.name
    state['constraint_id'] = constraint_id


def place_object(robot_id, held_spec, target_spec, state):
    """Place the held object at target_spec and release it."""
    if state.get('held_object') != held_spec.name:
        raise SkillExecutionError(
            'Cannot place "{}": the gripper is holding "{}".'.format(
                held_spec.name, state.get('held_object')
            )
        )

    if (
        target_spec.place_release_position is None
        or target_spec.place_rest_position is None
    ):
        raise SkillExecutionError(
            'No place location is defined for target "{}".'.format(
                target_spec.name
            )
        )

    constraint_id = state.get('constraint_id')

    move_to_position(robot_id, target_spec.above_position, duration=2.0)
    move_to_position(
        robot_id,
        target_spec.place_release_position,
        duration=1.5
    )

    if constraint_id is not None:
        p.removeConstraint(constraint_id)

    p.resetBasePositionAndOrientation(
        held_spec.body_id,
        target_spec.place_rest_position,
        [0, 0, 0, 1]
    )
    p.resetBaseVelocity(
        held_spec.body_id,
        linearVelocity=[0, 0, 0],
        angularVelocity=[0, 0, 0]
    )

    open_gripper(robot_id)
    move_to_position(
        robot_id,
        target_spec.above_position,
        duration=1.5
    )

    _restore_collision(robot_id, held_spec.body_id)

    # Let the released object settle.
    _sleep_steps(_PHYSICS_FREQUENCY)

    state['held_object'] = None
    state['constraint_id'] = None


def build_skill_handlers(robot_id, object_table, state):
    """Build the move_to/pick/place handler dict for a SkillRegistry."""

    def require_spec(name):
        spec = object_table.get(name)

        if spec is None:
            raise SkillExecutionError(
                'Unknown scene object "{}". Known: {}.'.format(
                    name, ', '.join(sorted(object_table.keys()))
                )
            )

        return spec

    def handle_move_to(action, context):
        spec = require_spec(action.object)
        move_to_position(robot_id, spec.above_position, duration=2.0)

    def handle_pick(action, context):
        spec = require_spec(action.object)
        pick_object(robot_id, spec, state)

    def handle_place(action, context):
        held_spec = require_spec(action.object)
        target_spec = require_spec(action.target)
        place_object(robot_id, held_spec, target_spec, state)

    return {
        'move_to': handle_move_to,
        'pick': handle_pick,
        'place': handle_place,
    }


def main():
    robot_id, object_table = create_scene()

    state = {'held_object': None, 'constraint_id': None}

    registry = SkillRegistry()

    for name, handler in build_skill_handlers(
        robot_id, object_table, state
    ).items():
        registry.register(name, handler)

    print('PyBullet scene started.')
    print('The robot will begin after two seconds.')

    _sleep_steps(2 * _PHYSICS_FREQUENCY)

    raw_actions = [
        {'skill': 'move_to', 'object': 'apple'},
        {'skill': 'pick', 'object': 'apple'},
        {'skill': 'move_to', 'object': 'basket'},
        {'skill': 'place', 'object': 'apple', 'target': 'basket'},
    ]

    for action in parse_actions(raw_actions):
        print('Executing skill: {}'.format(action.describe()))
        registry.execute(action)

    print('Pick-and-place skill sequence completed.')
    print('Close the PyBullet window or press Ctrl+C to stop.')

    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / _PHYSICS_FREQUENCY)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == '__main__':
    main()
