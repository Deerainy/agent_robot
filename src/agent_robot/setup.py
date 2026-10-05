import os
from glob import glob
from setuptools import find_packages, setup


package_name = 'agent_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
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
        ],
    },
)
