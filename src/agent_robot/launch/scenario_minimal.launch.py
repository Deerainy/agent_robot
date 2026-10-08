"""Minimal M5 stack: world truth, executor, recorder, optional planner."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    backend = LaunchConfiguration('backend')
    use_planner = LaunchConfiguration('use_planner')
    output_dir = LaunchConfiguration('output_dir')
    scenario_spec = LaunchConfiguration('scenario_spec')
    category = LaunchConfiguration('category')
    seed = LaunchConfiguration('seed')
    effective_category = PythonExpression([
        "'' if '", scenario_spec, "' else '", category, "'"
    ])
    effective_seed = PythonExpression([
        "-1 if '", scenario_spec, "' else ", seed
    ])
    scenario_id = LaunchConfiguration('scenario_id')
    mock_action_duration = LaunchConfiguration('mock_action_duration')
    pybullet_connection_mode = LaunchConfiguration(
        'pybullet_connection_mode'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'backend',
            default_value='mock',
            description='mock or pybullet (world + executor backend).',
        ),
        DeclareLaunchArgument(
            'use_planner',
            default_value='false',
            description=(
                'Start DeepSeek task_planner (oracle/manual otherwise).'
            ),
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='~/ros2_ws/experiments/runs',
            description='Trajectory JSON output directory.',
        ),
        DeclareLaunchArgument(
            'scenario_spec',
            default_value='',
            description='Path to SceneSpec JSON (overrides category+seed).',
        ),
        DeclareLaunchArgument(
            'category',
            default_value='recovery',
            description='Scenario category when scenario_spec is empty.',
        ),
        DeclareLaunchArgument(
            'seed',
            default_value='7',
            description='Scenario seed when scenario_spec is empty.',
        ),
        DeclareLaunchArgument(
            'scenario_id',
            default_value='',
            description='Optional scenario_id override.',
        ),
        DeclareLaunchArgument(
            'mock_action_duration',
            default_value='0.05',
            description='Mock backend sleep seconds per action.',
        ),
        DeclareLaunchArgument(
            'pybullet_connection_mode',
            default_value='GUI',
            description='PyBullet connection mode: GUI or DIRECT.',
        ),

        Node(
            package='agent_robot',
            executable='world_node',
            name='world_node',
            parameters=[
                {'backend': backend},
                {'scenario_spec': scenario_spec},
                {'category': effective_category},
                {'seed': effective_seed},
                {'scenario_id': scenario_id},
            ],
            output='screen',
        ),

        Node(
            package='agent_robot',
            executable='task_planner',
            name='task_planner',
            condition=IfCondition(use_planner),
            output='screen',
        ),

        Node(
            package='agent_robot',
            executable='task_executor',
            name='task_executor',
            parameters=[
                {'backend': backend},
                {'mock_action_duration': mock_action_duration},
                {'scenario_spec': scenario_spec},
                {'category': effective_category},
                {'seed': effective_seed},
                {'pybullet_connection_mode': pybullet_connection_mode},
            ],
            output='screen',
        ),

        Node(
            package='agent_robot',
            executable='trajectory_recorder',
            name='trajectory_recorder',
            parameters=[
                {'output_dir': output_dir},
            ],
            output='screen',
        ),
    ])
