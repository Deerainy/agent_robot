import time
import math

import pybullet as p
import pybullet_data


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


def create_scene():
    # 连接 PyBullet 图形界面
    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError('Failed to open the PyBullet GUI.')

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 调整观察视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.6,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.45, 0.0, 0.25]
    )

    # 地面和桌子
    p.loadURDF('plane.urdf')

    p.loadURDF(
        'table/table.urdf',
        basePosition=[0.5, 0.0, -0.65],
        useFixedBase=True
    )

    # Franka Panda 机械臂
    robot_id = p.loadURDF(
        'franka_panda/panda.urdf',
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True
    )

    # 设置机械臂初始姿态
    initial_joint_positions = [
        0.0,
        -0.45,
        0.0,
        -2.2,
        0.0,
        1.8,
        0.8
    ]

    for joint_index, joint_position in enumerate(
        initial_joint_positions
    ):
        p.resetJointState(
            robot_id,
            joint_index,
            joint_position
        )

    # 红色苹果
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
        basePosition=[0.55, -0.20, 0.06]
    )

    # 蓝色杯子：先用圆柱体表示
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

    p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cup_collision,
        baseVisualShapeIndex=cup_visual,
        basePosition=[0.55, 0.05, 0.06]
    )

    # 棕色篮子
    basket_color = [0.45, 0.22, 0.08, 1.0]

    # 篮子底部
    create_box(
        [0.70, 0.28, 0.025],
        [0.13, 0.10, 0.025],
        basket_color
    )

    # 四面篮筐
    create_box(
        [0.70, 0.18, 0.09],
        [0.13, 0.015, 0.09],
        basket_color
    )
    create_box(
        [0.70, 0.38, 0.09],
        [0.13, 0.015, 0.09],
        basket_color
    )
    create_box(
        [0.57, 0.28, 0.09],
        [0.015, 0.10, 0.09],
        basket_color
    )
    create_box(
        [0.83, 0.28, 0.09],
        [0.015, 0.10, 0.09],
        basket_color
    )

    return robot_id, apple_id

def move_end_effector(robot_id, target_position, duration=2.0):
    """通过逆运动学把机械臂末端移动到目标位置。"""
    end_effector_link = 11

    downward_orientation = p.getQuaternionFromEuler(
        [math.pi, 0.0, 0.0]
    )

    joint_targets = p.calculateInverseKinematics(
        robot_id,
        end_effector_link,
        target_position,
        downward_orientation,
        maxNumIterations=200,
        residualThreshold=1e-5
    )

    simulation_steps = int(duration * 240)

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
        time.sleep(1.0 / 240.0)


def control_gripper(robot_id, opening, duration=0.7):
    """控制 Panda 两个夹爪手指。"""
    finger_joints = [9, 10]
    simulation_steps = int(duration * 240)

    for _ in range(simulation_steps):
        for joint_index in finger_joints:
            p.setJointMotorControl2(
                robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=opening,
                force=100
            )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def attach_apple(robot_id, apple_id):
    """保持苹果当前位置，并固定到机械臂末端。"""
    end_effector_link = 11

    link_state = p.getLinkState(
        robot_id,
        end_effector_link,
        computeForwardKinematics=True
    )

    apple_position, apple_orientation = (
        p.getBasePositionAndOrientation(apple_id)
    )

    inverse_position, inverse_orientation = p.invertTransform(
        link_state[4],
        link_state[5]
    )

    relative_position, relative_orientation = p.multiplyTransforms(
        inverse_position,
        inverse_orientation,
        apple_position,
        apple_orientation
    )

    # 抓住后关闭苹果与机械臂之间的碰撞，避免夹爪把苹果弹飞
    for link_index in range(-1, p.getNumJoints(robot_id)):
        p.setCollisionFilterPair(
            robot_id,
            apple_id,
            link_index,
            -1,
            enableCollision=0
        )

    return p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=end_effector_link,
        childBodyUniqueId=apple_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=relative_position,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=relative_orientation,
        childFrameOrientation=[0, 0, 0, 1]
    )

def run_pick(robot_id, apple_id):
    """抓起苹果并保持在空中。"""
    apple_above = [0.55, -0.20, 0.32]
    apple_grasp = [0.55, -0.20, 0.085]

    print('Pick Step 1: Opening gripper')
    control_gripper(robot_id, 0.04)

    print('Pick Step 2: Moving above apple')
    move_end_effector(
        robot_id,
        apple_above,
        duration=2.0
    )

    print('Pick Step 3: Moving down to apple')
    move_end_effector(
        robot_id,
        apple_grasp,
        duration=1.5
    )

    print('Pick Step 4: Closing gripper')
    control_gripper(robot_id, 0.0)

    constraint_id = attach_apple(
        robot_id,
        apple_id
    )

    print('Pick Step 5: Lifting apple')
    move_end_effector(
        robot_id,
        apple_above,
        duration=1.5
    )

    print('Apple picked successfully.')

    return constraint_id

def run_pick_and_place(robot_id, apple_id):
    """执行抓取苹果并放入篮子的演示。"""
    apple_above = [0.55, -0.20, 0.32]
    apple_grasp = [0.55, -0.20, 0.085]

    basket_above = [0.70, 0.28, 0.38]
    basket_release = [0.70, 0.28, 0.14]

    print('Step 1: Opening gripper')
    control_gripper(robot_id, 0.04)

    print('Step 2: Moving above apple')
    move_end_effector(robot_id, apple_above)

    print('Step 3: Moving down to apple')
    move_end_effector(robot_id, apple_grasp)

    print('Step 4: Closing gripper')
    control_gripper(robot_id, 0.0)

    constraint_id = attach_apple(
        robot_id,
        apple_id
    )

    print('Step 5: Lifting apple')
    move_end_effector(robot_id, apple_above)

    print('Step 6: Moving above basket')
    move_end_effector(robot_id, basket_above)

    print('Step 7: Lowering apple into basket')
    move_end_effector(robot_id, basket_release)

    print('Step 8: Releasing apple')

    # 解除机械臂与苹果的固定关系
    p.removeConstraint(constraint_id)

    # 将苹果稳定放在篮子内部，消除约束解除时的误差和残余速度
    p.resetBasePositionAndOrientation(
        apple_id,
        [0.70, 0.28, 0.105],
        [0, 0, 0, 1]
    )

    p.resetBaseVelocity(
        apple_id,
        linearVelocity=[0, 0, 0],
        angularVelocity=[0, 0, 0]
    )

    # 张开夹爪
    control_gripper(robot_id, 0.04)

    # 在机械臂与苹果碰撞关闭的情况下向上撤离
    move_end_effector(
        robot_id,
        basket_above,
        duration=1.5
    )

    # 机械臂离开后，恢复苹果与机械臂的碰撞
    for link_index in range(-1, p.getNumJoints(robot_id)):
        p.setCollisionFilterPair(
            robot_id,
            apple_id,
            link_index,
            -1,
            enableCollision=1
        )

    # 等待场景稳定
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    print('Pick-and-place task completed.')

def main():
    robot_id, apple_id = create_scene()
    p.changeDynamics(
        apple_id,
        -1,
        lateralFriction=0.9,
        rollingFriction=0.05,
        spinningFriction=0.05,
        restitution=0.0
    )

    print('PyBullet scene started.')
    print('The robot will begin after two seconds.')

    for _ in range(480):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    run_pick_and_place(robot_id, apple_id)

    print('Close the PyBullet window or press Ctrl+C to stop.')

    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == '__main__':
    main()