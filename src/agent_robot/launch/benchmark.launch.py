"""Launch the batch harness for a generated M5 episode directory."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'input_dir',
            default_value='datasets/m5_benchmark',
            description='Directory created by generate_benchmark_episodes.',
        ),
        DeclareLaunchArgument(
            'results_path',
            default_value='experiments/m5_benchmark/benchmark_results.json',
            description='Summary output JSON path.',
        ),
        DeclareLaunchArgument(
            'planner_mode',
            default_value='mock',
            description='mock for deterministic offline planning or deepseek.',
        ),
        DeclareLaunchArgument(
            'vision_backend',
            default_value='synthetic_opencv',
            description=(
                'synthetic_opencv for generated images, opencv for photos, '
                'or deepseek VLM.'
            ),
        ),
        DeclareLaunchArgument(
            'timeout',
            default_value='90',
            description='Per-episode timeout in seconds.',
        ),
        DeclareLaunchArgument(
            'ros_domain_id',
            default_value='auto',
            description=(
                'Isolated DDS domain; use auto or an ID from 0 to 232.'
            ),
        ),
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'agent_robot', 'run_benchmark',
                '--input-dir', LaunchConfiguration('input_dir'),
                '--output', LaunchConfiguration('results_path'),
                '--planner-mode', LaunchConfiguration('planner_mode'),
                '--vision-backend', LaunchConfiguration('vision_backend'),
                '--timeout', LaunchConfiguration('timeout'),
                '--ros-domain-id', LaunchConfiguration('ros_domain_id'),
            ],
            output='screen',
        ),
    ])
