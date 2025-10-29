from setuptools import setup
from glob import glob
import os

package_name = 'maze_nav_system'

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
    maintainer='Kenneth Martinez',
    maintainer_email='kennethaldahir.martinezmoreno@ontariotechu.net',
    description='Maze navigation with opencv',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           'edge_node = maze_nav_system.edge_detection:main',
        ],
    },
)
