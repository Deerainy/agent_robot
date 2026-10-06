from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    backend = LaunchConfiguration('backend')

    return LaunchDescription([
        DeclareLaunchArgument(
            'backend',
            default_value='pybullet',
            description=(
                'Skill execution backend used by task_executor '
                '(mock or pybullet).'
            )
        ),

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
            parameters=[
                {'backend': backend}
            ],
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
            executable='trajectory_recorder',
            name='trajectory_recorder',
            output='screen'
        ),
    ])
