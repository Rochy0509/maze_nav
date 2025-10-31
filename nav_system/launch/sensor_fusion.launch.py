from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    nav_system_dir = get_package_share_directory('nav_system')
    ekf_config = os.path.join(nav_system_dir, 'config', 'ekf.yaml')
    return LaunchDescription([
        
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_component',
            name='imu_filter',
            output='screen',
            parameters=[{
                'use_mag': True,
                'publish_tf': False,
                'world_frame': 'enu',
                'gain': 0.01,
                'zeta': 0.0,
                'fixed_frame': 'odom',
                'imu_topic': '/imu_raw',
                'mag_topic': '/imu/mag',
            }]
        ),
        
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                ekf_config,
                {'use_sim_time': False}
            ],
            remappings=[
                ('odometry/filtered', '/odom')  
            ]
        )
    ])