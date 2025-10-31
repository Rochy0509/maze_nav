from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    nav_system_dir = get_package_share_directory('nav_system')
    ekf_config = os.path.join(nav_system_dir, 'config', 'ekf.yaml') 
        
    return LaunchDescription([

        Node(package='nav_system', executable='serial_bridge', name='serial_bridge'),
        
        # IMU filter
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_component',
            parameters=[{'use_mag': True, 'world_frame': 'enu'}]
        ),
        
        # EKF
        Node(
            package='robot_localization',
            executable='ekf_node',
            parameters=[ekf_config]
        ),
        
        Node(
            package='nav_system',
            executable='robot_controller',
            name='robot_controller'
        )
])