"""Launch perception-grounded planning, PyBullet execution and recording."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_path = LaunchConfiguration('image_path')
    results_dir = LaunchConfiguration('results_dir')
    output_dir = LaunchConfiguration('output_dir')
    world_x_min = LaunchConfiguration('world_x_min')
    world_x_max = LaunchConfiguration('world_x_max')
    world_y_min = LaunchConfiguration('world_y_min')
    world_y_max = LaunchConfiguration('world_y_max')
    roi_left = LaunchConfiguration('roi_left')
    roi_top = LaunchConfiguration('roi_top')
    roi_right = LaunchConfiguration('roi_right')
    roi_bottom = LaunchConfiguration('roi_bottom')
    connection_mode = LaunchConfiguration('pybullet_connection_mode')
    planner_mode = LaunchConfiguration('planner_mode')
    vision_backend = LaunchConfiguration('vision_backend')

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_path',
            default_value='',
            description=(
                'Optional image to process automatically on startup; '
                'otherwise publish a path to /image_path.'
            ),
        ),
        DeclareLaunchArgument(
            'results_dir',
            default_value='results',
            description='Directory for image-derived scene graph JSON.',
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='experiments/vision_runs',
            description='Directory for execution trajectory JSON.',
        ),
        DeclareLaunchArgument(
            'world_x_min',
            default_value='0.38',
            description='Minimum calibrated tabletop X in meters.',
        ),
        DeclareLaunchArgument(
            'world_x_max',
            default_value='0.76',
            description='Maximum calibrated tabletop X in meters.',
        ),
        DeclareLaunchArgument(
            'world_y_min',
            default_value='-0.28',
            description='Minimum calibrated tabletop Y in meters.',
        ),
        DeclareLaunchArgument(
            'world_y_max',
            default_value='0.28',
            description='Maximum calibrated tabletop Y in meters.',
        ),
        DeclareLaunchArgument(
            'roi_left',
            default_value='0.0',
            description='Normalized image ROI left edge.',
        ),
        DeclareLaunchArgument(
            'roi_top',
            default_value='0.0',
            description='Normalized image ROI top edge.',
        ),
        DeclareLaunchArgument(
            'roi_right',
            default_value='1.0',
            description='Normalized image ROI right edge.',
        ),
        DeclareLaunchArgument(
            'roi_bottom',
            default_value='1.0',
            description='Normalized image ROI bottom edge.',
        ),
        DeclareLaunchArgument(
            'pybullet_connection_mode',
            default_value='GUI',
            description='PyBullet GUI or DIRECT mode.',
        ),
        DeclareLaunchArgument(
            'planner_mode',
            default_value='deepseek',
            description='deepseek for LLM planning; mock for offline testing.',
        ),
        DeclareLaunchArgument(
            'vision_backend',
            default_value=EnvironmentVariable(
                'VISION_BACKEND',
                default_value='deepseek',
            ),
            description='deepseek for VLM perception or opencv locally.',
        ),
        Node(
            package='agent_robot',
            executable='perception_node',
            name='perception_node',
            parameters=[{
                'image_path': image_path,
                'results_dir': results_dir,
                'vision_backend': vision_backend,
                'world_x_min': world_x_min,
                'world_x_max': world_x_max,
                'world_y_min': world_y_min,
                'world_y_max': world_y_max,
                'roi_left': roi_left,
                'roi_top': roi_top,
                'roi_right': roi_right,
                'roi_bottom': roi_bottom,
            }],
            output='screen',
        ),
        Node(
            package='agent_robot',
            executable='task_planner',
            name='task_planner',
            parameters=[{'planner_mode': planner_mode}],
            output='screen',
        ),
        Node(
            package='agent_robot',
            executable='task_executor',
            name='task_executor',
            parameters=[{
                'backend': 'pybullet',
                'pybullet_connection_mode': connection_mode,
                'require_perception_scene': True,
            }],
            output='screen',
        ),
        Node(
            package='agent_robot',
            executable='trajectory_recorder',
            name='trajectory_recorder',
            parameters=[{'output_dir': output_dir}],
            output='screen',
        ),
    ])
