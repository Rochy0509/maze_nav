#!/usr/bin/env python3
"""
Launch edge_detection and navigation nodes together.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('maze_nav_system')
    default_params = os.path.join(pkg_share, 'config', 'edge_detection.yaml')

    # Launch-time knobs for edge detection
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Full path to the edge_detection YAML config'
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    publish_compressed_arg = DeclareLaunchArgument(
        'publish_compressed',
        default_value='false',
        description='Publish /edge_detection/annotated/compressed (jpeg)'
    )

    # Launch the edge detection node
    edge_detection_node = Node(
        package='maze_nav_system',
        executable='edge_node',
        name='edge_detection',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),                         # load YAML
            {'input_topic': LaunchConfiguration('image_topic')},        # optional override
            {'publish_compressed': LaunchConfiguration('publish_compressed')},
        ],
    )

    # Launch the navigation node
    navigation_node = Node(
        package='maze_nav_system',
        executable='navigation_node', 
        name='navigation_node',
        output='screen',
        parameters=[
            # Example: {'linear_speed': 0.2, 'angular_speed': 0.5}
        ],
    )

    return LaunchDescription([
        params_file_arg,
        image_topic_arg,
        publish_compressed_arg,
        edge_detection_node,
        navigation_node,
    ])