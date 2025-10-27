from setuptools import setup
from glob import glob
import os

package_name = 'maze_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tron',
    maintainer_email='tron@tron.local',
    description='Maze navigation with RRT* and pure pursuit',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'maze_planner = maze_nav.maze_planner:main',
            'swerve_controller = maze_nav.swerve_controller:main',
            'send_goal = maze_nav.send_goal:main',
	    'arduino_swerve_bridge = maze_nav.arduino_swerve_bridge:main',
        ],
    },
)
