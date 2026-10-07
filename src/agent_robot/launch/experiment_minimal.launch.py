"""Headless minimal stack for M4 comparative experiments.

Nodes: task_executor (mock backend), trajectory_recorder, and optionally
task_planner (disabled in replay mode). No vision / GUI / visualizer /
environment_node: the executor uses its catalog-default scene graph
(apple/cup/basket on the table) so execution-applied state is never
overwritten by periodic perception.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    backend = LaunchConfiguration('backend')
    use_planner = LaunchConfiguration('use_planner')
    output_dir = LaunchConfiguration('output_dir')

    return LaunchDescription([
        DeclareLaunchArgument(
            'backend',
            default_value='mock',
            description='Skill execution backend (experiments use mock).'
        ),
        DeclareLaunchArgument(
            'use_planner',
            default_value='true',
            description='Start task_planner (false in replay mode).'
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='~/ros2_ws/experiments/runs',
            description='Per-task trajectory output directory.'
        ),

        Node(
            package='agent_robot',
            executable='task_planner',
            name='task_planner',
            condition=IfCondition(use_planner),
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
            executable='trajectory_recorder',
            name='trajectory_recorder',
            parameters=[
                {'output_dir': output_dir}
            ],
            output='screen'
        ),
    ])
