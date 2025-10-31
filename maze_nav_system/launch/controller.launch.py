from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    maze_nav_system_dir = get_package_share_directory('maze_nav_system')
    ekf_config = os.path.join(maze_nav_system_dir, 'config', 'ekf.yaml')

    return LaunchDescription([

        # Serial bridge node
        Node(
            package='maze_nav_system',
            executable='serial_bridge',
            name='serial_bridge',
            output='screen'
        ),

        # IMU filter (Madgwick)
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter',
            output='screen',
            parameters=[{
                'use_mag': True,
                'publish_tf': False,
                'world_frame': 'enu',
                'imu_topic': '/imu/raw',   
                'mag_topic': '/imu/mag',  
            }]
        ),

        # EKF for localization
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config]
        ),

        # Robot controller node
        Node(
            package='maze_nav_system',
            executable='robot_controller',
            name='robot_controller',
            output='screen'
        ),
    ])
