#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('maze_nav')
    
    # Config file
    edge_config = os.path.join(pkg_share, 'config', 'cv_params.yaml')
    
    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        
        # Edge Detection Node
        Node(
            package='maze_nav',
            executable='edge_node',
            name='edge_detection_node',
            parameters=[
                edge_config,
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            remappings=[],
            output='screen'
        ),
    ])