import os
from glob import glob
from setuptools import setup


package_name = 'agent_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[
        package_name,
        package_name + '.experiments',
        package_name + '.perception',
        package_name + '.scenarios',
    ],
    package_data={
        package_name + '.experiments': ['*.json'],
        package_name + '.scenarios': ['*.json'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            'share/' + package_name + '/images',
            glob('images/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deerainy',
    maintainer_email='yuxinlu0410@gmail.com',
    description=(
        'Multimodal ROS 2 agent with visual perception, '
        'LLM task planning and automatic replanning.'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_planner = agent_robot.task_planner:main',
            'task_executor = agent_robot.task_executor:main',
            'environment_node = agent_robot.environment_node:main',
            'vision_node = agent_robot.vision_node:main',
            'world_node = agent_robot.world_node:main',
            'scene_visualizer = agent_robot.scene_visualizer:main',
            'scene_loader = agent_robot.scene_loader:main',
            'trajectory_recorder = agent_robot.trajectory_recorder:main',
            'score_trajectories = agent_robot.quality_scorer:main',
            'run_experiments = agent_robot.experiments.batch_runner:main',
            'run_baseline = agent_robot.experiments.baseline_runner:main',
            'report_experiments = agent_robot.experiments.report:main',
            'generate_vision_dataset = agent_robot.scenarios.dataset:main',
            'generate_benchmark_episodes = '
            'agent_robot.scenarios.benchmark:main',
            'run_benchmark = '
            'agent_robot.experiments.benchmark_runner:main',
            'perception_node = agent_robot.perception.perception_node:main',
            'web_demo = agent_robot.web_demo:main',
            'evaluate_perception = '
            'agent_robot.perception.evaluate_dataset:main',
        ],
    },
)
