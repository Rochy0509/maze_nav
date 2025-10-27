#!/usr/bin/env python3
"""Swerve drive controller"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np

class SwerveController(Node):
    def __init__(self):
        super().__init__('swerve_controller')
        
        self.declare_parameter('robot.wheel_base_m', 0.146)
        self.declare_parameter('robot.wheel_radius_m', 0.031)
        
        self.wheel_base = self.get_parameter('robot.wheel_base_m').value
        
        self.swerve_pub = self.create_publisher(Float32MultiArray, 'swerve_commands', 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        self.get_logger().info('Swerve Controller ready')
    
    def cmd_vel_callback(self, msg: Twist):
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # Differential swerve kinematics
        left_vx = linear_x - (angular_z * self.wheel_base / 2.0)
        right_vx = linear_x + (angular_z * self.wheel_base / 2.0)
        
        left_speed = abs(left_vx)
        left_angle = 0.0 if left_vx >= 0 else np.pi
        
        right_speed = abs(right_vx)
        right_angle = 0.0 if right_vx >= 0 else np.pi
        
        cmd = Float32MultiArray()
        cmd.data = [float(left_angle), float(left_speed), 
                   float(right_angle), float(right_speed)]
        self.swerve_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SwerveController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
