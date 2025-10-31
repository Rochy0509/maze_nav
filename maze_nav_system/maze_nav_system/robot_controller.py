#!/usr/bin/env python3
# robot_controller.py

import math
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class RobotController(Node):
    def __init__(self, node_name: str = 'robot_controller'):
        super().__init__(node_name)

        # --- Gains (declare + read) ---
        # Defaults chosen to be reasonable; adjust in launch or via params.
        self.k_p_linear_: float = self.declare_parameter('k_p_linear', 0.6).value
        self.k_p_angular_: float = self.declare_parameter('k_p_angular', 1.2).value

        # Safety caps (since C++ snippet references these, mirror them here)
        self.max_linear_speed_: float = 0.3     # m/s
        self.max_angular_speed_: float = 1.0    # rad/s

        # --- Publishers ---
        self.cmd_vel_pub_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # --- Subscribers ---
        self.current_x_: float = 0.0
        self.current_y_: float = 0.0
        self.current_yaw_: float = 0.0

        self.odom_sub_ = self.create_subscription(
            Odometry,
            '/odom_filtered',
            self.odom_callback,
            10
        )

        self.imu_sub_ = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10
        )

        # --- Target state ---
        self.has_target_: bool = False
        self.target_x_: float = 0.0
        self.target_y_: float = 0.0
        self.target_yaw_: float = 0.0
        self._first_run_: bool = True

        # --- Control loop timer (20 Hz) ---
        self.timer_ = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info('Robot controller started')

    # ----------------- Callbacks -----------------

    def odom_callback(self, msg: Odometry) -> None:
        self.current_x_ = msg.pose.pose.position.x
        self.current_y_ = msg.pose.pose.position.y

    def imu_callback(self, msg: Imu) -> None:
        self.current_yaw_ = self.get_yaw(msg.orientation)

    # ----------------- Helpers -----------------

    @staticmethod
    def get_yaw(q: Quaternion) -> float:
        """
        Extract yaw (Z rotation) from a geometry_msgs/Quaternion.
        Equivalent to tf2::Matrix3x3(q).getRPY(...).
        """
        # Convert quaternion to yaw using standard formulas
        # yaw (z-axis rotation) from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def timer_callback(self) -> None:
        # Initialize a hardcoded target once, just like the C++ example.
        if self._first_run_:
            self.target_x_ = 1.0     # go 1 m forward
            self.target_y_ = 0.0
            self.target_yaw_ = 0.0   # face forward
            self.has_target_ = True
            self._first_run_ = False

        if not self.has_target_:
            return

        # --- Compute position errors ---
        dx = self.target_x_ - self.current_x_
        dy = self.target_y_ - self.current_y_
        distance_error = math.sqrt(dx * dx + dy * dy)

        # Desired heading toward the target
        desired_yaw = math.atan2(dy, dx)
        heading_error = self._normalize_angle(desired_yaw - self.current_yaw_)

        cmd = Twist()

        if distance_error > 0.05:  # 5 cm tolerance
            # Move toward target
            cmd.linear.x = min(self.k_p_linear_ * distance_error, self.max_linear_speed_)
            raw_w = self.k_p_angular_ * heading_error
            cmd.angular.z = max(-self.max_angular_speed_, min(raw_w, self.max_angular_speed_))
        else:
            # Hold (correct) final heading
            final_heading_error = self._normalize_angle(self.target_yaw_ - self.current_yaw_)
            raw_w = self.k_p_angular_ * final_heading_error
            cmd.angular.z = max(-self.max_angular_speed_, min(raw_w, self.max_angular_speed_))

        self.cmd_vel_pub_.publish(cmd)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
