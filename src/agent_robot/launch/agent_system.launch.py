from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='agent_robot',
            executable='vision_node',
            name='vision_node',
            output='screen'
        ),

        Node(
            package='agent_robot',
            executable='task_planner',
            name='task_planner',
            output='screen'
        ),

        Node(
            package='agent_robot',
            executable='task_executor',
            name='task_executor',
            output='screen'
        ),
        Node(
            package='agent_robot',
            executable='scene_visualizer',
            name='scene_visualizer',
            output='screen'
        ),
        Node(
            package='agent_robot',
            executable='scene_loader',
            name='scene_loader',
            output='screen'
        ),
        Node(
            package='agent_robot',
            executable='pybullet_bridge',
            name='pybullet_bridge',
            output='screen'
        ),
    ])