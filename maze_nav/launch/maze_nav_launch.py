#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('maze_nav')
    
    # Config files
    map_config = os.path.join(pkg_share, 'config', 'map_config.yaml')
    ekf_config = os.path.join(pkg_share, 'config', 'ekf.yaml')
    
    # Parameters as dictionary (more reliable than YAML files for simple params)
    robot_params = {
        'robot.wheel_base_m': 0.146,
        'robot.wheel_radius_m': 0.031,
        'robot.robot_radius_m': 0.15,
    }
    
    map_params = {
        'map.image_path': '/home/tron/maze_maps/test_maze.png',
        'map.real_width_m': 3.658,
        'map.real_height_m': 3.658,
        'map.robot_radius_m': 0.15,
        'planner.max_iterations': 1500,
        'planner.step_size_m': 0.08,
        'planner.goal_tolerance_m': 0.12,
        'controller.lookahead_distance_m': 0.25,
        'controller.max_linear_speed_ms': 0.4,
        'controller.max_angular_speed_rads': 1.5,
    }
    
    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        
        # EKF for sensor fusion (only if ekf.yaml exists and is correct)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            parameters=[
                ekf_config if os.path.exists(ekf_config) else {},
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            remappings=[
                ('odometry/filtered', 'odom_filtered')
            ],
            output='screen'
        ),
        
        # Maze navigator (RRT* + Pure Pursuit)
        Node(
            package='maze_nav',
            executable='maze_planner',
            name='maze_navigator',
            parameters=[
                map_params,
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        ),
        
        # Swerve controller
        Node(
            package='maze_nav',
            executable='swerve_controller',
            name='swerve_controller',
            parameters=[
                robot_params,
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        ),
    ])
