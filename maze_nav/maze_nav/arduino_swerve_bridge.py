#!/usr/bin/env python3
"""
ROS2 Bridge for Arduino Swerve Drive with JSON Protocol
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
import serial
import json
import math
import time

class ArduinoSwerveBridge(Node):
    def __init__(self):
        super().__init__('arduino_swerve_bridge')
        
        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_base_m', 0.146)
        self.declare_parameter('wheel_radius_m', 0.031)
        self.declare_parameter('encoder_cpr', 2880)
        
        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        self.wheel_base = self.get_parameter('wheel_base_m').value
        self.wheel_radius = self.get_parameter('wheel_radius_m').value
        self.encoder_cpr = self.get_parameter('encoder_cpr').value
        
        # Connect to Arduino
        try:
            self.serial = serial.Serial(port, baud, timeout=0.01)
            time.sleep(2)
            self.get_logger().info(f'Connected to Arduino on {port}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect: {e}')
            raise
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        
        # Subscriber
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # State
        self.last_left_count = 0
        self.last_right_count = 0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_odom_time = self.get_clock().now()
        
        # Current swerve angles (for display/debugging)
        self.left_steer_angle = 0.0
        self.right_steer_angle = 0.0
        
        # Timer for reading serial
        self.create_timer(0.001, self.read_serial)
        
        self.get_logger().info('Arduino Swerve Bridge ready')
    
    def read_serial(self):
        """Read JSON from Arduino"""
        try:
            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8').strip()
                if line.startswith('{'):
                    try:
                        data = json.loads(line)
                        self.process_message(data)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            self.get_logger().warn(f'Serial error: {e}')
    
    def process_message(self, data):
        """Process JSON from Arduino"""
        if 'imu' in data:
            self.handle_imu(data['imu'])
        elif 'enc' in data:
            self.handle_encoders(data['enc'])
        elif 'status' in data:
            self.get_logger().info(f"Arduino: {data['status']}")
        elif 'error' in data:
            self.get_logger().error(f"Arduino: {data['error']}")
        elif 'calib' in data:
            self.get_logger().info(f"Steering calibrated: {data['calib']}")
    
    def handle_imu(self, imu_data):
        """Handle IMU: [roll, pitch, yaw] in degrees"""
        if len(imu_data) != 3:
            return
        
        roll, pitch, yaw = [math.radians(x) for x in imu_data]
        
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        # Convert to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy
        
        msg.orientation_covariance[0] = 0.01
        msg.orientation_covariance[4] = 0.01
        msg.orientation_covariance[8] = 0.02
        
        self.imu_pub.publish(msg)
    
    def handle_encoders(self, enc_data):
        """
        Handle encoders: [left_drive_count, right_drive_count, left_angle_rad, right_angle_rad]
        Calculate odometry from drive encoder counts
        """
        if len(enc_data) != 4:
            return
        
        left_count = enc_data[0]
        right_count = enc_data[1]
        self.left_steer_angle = enc_data[2]
        self.right_steer_angle = enc_data[3]
        
        current_time = self.get_clock().now()
        dt = (current_time - self.last_odom_time).nanoseconds / 1e9
        
        if dt <= 0.0 or dt > 1.0:
            self.last_left_count = left_count
            self.last_right_count = right_count
            self.last_odom_time = current_time
            return
        
        # Calculate wheel movement
        delta_left = left_count - self.last_left_count
        delta_right = right_count - self.last_right_count
        
        left_rotations = delta_left / self.encoder_cpr
        right_rotations = delta_right / self.encoder_cpr
        
        left_distance = left_rotations * 2 * math.pi * self.wheel_radius
        right_distance = right_rotations * 2 * math.pi * self.wheel_radius
        
        # Differential drive odometry
        forward_distance = (left_distance + right_distance) / 2.0
        delta_theta = (right_distance - left_distance) / self.wheel_base
        
        # Update pose
        self.odom_theta += delta_theta
        self.odom_x += forward_distance * math.cos(self.odom_theta)
        self.odom_y += forward_distance * math.sin(self.odom_theta)
        
        # Velocities
        linear_vel = forward_distance / dt
        angular_vel = delta_theta / dt
        
        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        odom_msg.pose.pose.position.x = self.odom_x
        odom_msg.pose.pose.position.y = self.odom_y
        odom_msg.pose.pose.orientation.w = math.cos(self.odom_theta / 2.0)
        odom_msg.pose.pose.orientation.z = math.sin(self.odom_theta / 2.0)
        
        odom_msg.twist.twist.linear.x = linear_vel
        odom_msg.twist.twist.angular.z = angular_vel
        
        odom_msg.pose.covariance[0] = 0.001
        odom_msg.pose.covariance[7] = 0.001
        odom_msg.pose.covariance[35] = 0.01
        odom_msg.twist.covariance[0] = 0.001
        odom_msg.twist.covariance[35] = 0.01
        
        self.odom_pub.publish(odom_msg)
        
        # Publish TF
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.odom_x
        t.transform.translation.y = self.odom_y
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)
        
        self.last_left_count = left_count
        self.last_right_count = right_count
        self.last_odom_time = current_time
    
    def cmd_vel_callback(self, msg: Twist):
        """
        Convert Twist to swerve commands
        Send: {"swerve":[left_angle, left_speed, right_angle, right_speed]}
        """
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # Differential swerve kinematics
        left_vx = linear_x - (angular_z * self.wheel_base / 2.0)
        right_vx = linear_x + (angular_z * self.wheel_base / 2.0)
        
        # Convert to magnitude and angle
        left_speed = abs(left_vx)
        left_angle = 0.0 if left_vx >= 0 else math.pi
        
        right_speed = abs(right_vx)
        right_angle = 0.0 if right_vx >= 0 else math.pi
        
        # Send to Arduino
        cmd = {
            "swerve": [
                round(left_angle, 3),
                round(left_speed, 3),
                round(right_angle, 3),
                round(right_speed, 3)
            ]
        }
        
        try:
            json_str = json.dumps(cmd) + '\n'
            self.serial.write(json_str.encode('utf-8'))
        except Exception as e:
            self.get_logger().warn(f'Failed to send: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ArduinoSwerveBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
