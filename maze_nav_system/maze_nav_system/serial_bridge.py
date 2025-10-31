#!/usr/bin/env python3
# serial_bridge.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, Range, MagneticField
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor

import serial
import json
import math
from typing import Optional


def get_baud_rate(rate: int) -> int:
    supported = {
        9600, 19200, 38400, 57600, 115200, 230400, 460800, 500000,
        576000, 921600, 1000000, 1152000, 1500000, 2000000, 2500000,
        3000000, 3500000, 4000000
    }
    if rate not in supported:
        raise ValueError(f"Unsupported baud rate: {rate}")
    return rate


class SerialBridge(Node):
    def __init__(self):
        super().__init__('arduino_bridge')

        # Parameters
        self.declare_parameter(
            'serial_port', '/dev/ttyACM0',
            ParameterDescriptor(description='Serial device path'))
        self.declare_parameter(
            'baud_rate', 115200,
            ParameterDescriptor(description='Serial baud rate'))

        port_name: str = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate_int: int = self.get_parameter('baud_rate').get_parameter_value().integer_value

        # Serial init
        self.serial: Optional[serial.Serial] = None
        try:
            baud = get_baud_rate(int(baud_rate_int))
            self.serial = serial.Serial(
                port=port_name,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                timeout=1.0  # non-blocking
            )
            self.get_logger().info(f'Serial port opened: {port_name} at {baud} baud')
        except ValueError as e:
            self.get_logger().error(f'Baud rate error: {e}')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')

        # Publishers
        self.imu_raw_pub_ = self.create_publisher(Imu, 'imu/raw', 10)
        self.odom_raw_pub_ = self.create_publisher(Odometry, 'odom', 10)
        self.tof_front_pub_ = self.create_publisher(Range, 'tof/front', 10)
        self.tof_left_pub_ = self.create_publisher(Range, 'tof/left', 10)
        self.tof_right_pub_ = self.create_publisher(Range, 'tof/right', 10)
        self.mag_pub_ = self.create_publisher(MagneticField, 'imu/mag', 10)

        # Timer for reading serial (every 10 ms)
        self.timer_ = self.create_timer(0.010, self.timer_callback)

        # Subscription for cmd_vel -> JSON out over serial
        self.cmd_vel_sub_ = self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_cb, 10
        )

        # Internal state / buffers
        self._rx_buffer = ''
        self._first_run = True
        self._last_left_rot = 0.0
        self._last_right_rot = 0.0
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

    # ---- Callbacks ----

    def _cmd_vel_cb(self, msg: Twist):
        if not self.serial or not self.serial.is_open:
            self.get_logger().warn('Serial not open!')  # ADD THIS
            return
        try:
            cmd_str = f'{{"cmd":[{msg.linear.x:.3f},{msg.linear.y:.3f},{msg.angular.z:.3f}]}}\n'
            self.get_logger().info(f'TX: {cmd_str.strip()}')  # ADD THIS - see what's being sent
            self.serial.write(cmd_str.encode('utf-8'))
            self.serial.flush()  # ADD THIS - ensure it's sent immediately
        except Exception as e:
            self.get_logger().warn(f'Serial write error: {e}')

    def timer_callback(self):
        if not self.serial or not self.serial.is_open:
            return
        try:
            while True:
                raw = self.serial.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', errors='ignore').strip()  # trims blanks
                if not line or not line.startswith('{'):
                    # ignore whitespace, boot messages, or any non-JSON line
                    continue
                try:
                    self.process_line(line)
                except json.JSONDecodeError as e:
                    self.get_logger().warn(f"JSON drop (not valid): {e.msg}")
        except Exception as e:
            self.get_logger().warn(f"Serial read error: {e}")

    # ---- Helpers ----

    def _now(self):
        return self.get_clock().now().to_msg()

    def process_line(self, line: str):
        if not line:
            return

        try:
            js = json.loads(line)

            # ---- IMU raw + mag ----
            if 'imu_raw' in js:
                arr = js['imu_raw']
                if isinstance(arr, (list, tuple)) and len(arr) >= 9:
                    ax = float(arr[0]) * 9.80665
                    ay = float(arr[1]) * 9.80665
                    az = float(arr[2]) * 9.80665

                    gx = float(arr[3]) * math.pi / 180.0
                    gy = float(arr[4]) * math.pi / 180.0
                    gz = float(arr[5]) * math.pi / 180.0

                    mx = float(arr[6])
                    my = float(arr[7])
                    mz = float(arr[8])

                    imu_msg = Imu()
                    imu_msg.header.stamp = self._now()
                    imu_msg.header.frame_id = 'imu_link'

                    # Unknown orientation
                    imu_msg.orientation.w = 1.0
                    imu_msg.orientation.x = 0.0
                    imu_msg.orientation.y = 0.0
                    imu_msg.orientation.z = 0.0
                    imu_msg.orientation_covariance = [-1.0] * 9

                    imu_msg.angular_velocity.x = gx
                    imu_msg.angular_velocity.y = gy
                    imu_msg.angular_velocity.z = gz
                    imu_msg.angular_velocity_covariance = [-1.0] * 9

                    imu_msg.linear_acceleration.x = ax
                    imu_msg.linear_acceleration.y = ay
                    imu_msg.linear_acceleration.z = az
                    imu_msg.linear_acceleration_covariance = [-1.0] * 9

                    self.imu_raw_pub_.publish(imu_msg)

                    mag_msg = MagneticField()
                    mag_msg.header.stamp = self._now()
                    mag_msg.header.frame_id = 'imu_link'
                    # microtesla -> tesla
                    mag_msg.magnetic_field.x = mx * 1e-6
                    mag_msg.magnetic_field.y = my * 1e-6
                    mag_msg.magnetic_field.z = mz * 1e-6
                    mag_msg.magnetic_field_covariance = [-1.0] * 9
                    self.mag_pub_.publish(mag_msg)

                    self.get_logger().debug(
                        f'IMU: a=({ax:.2f},{ay:.2f},{az:.2f}) '
                        f'g=({gx:.2f},{gy:.2f},{gz:.2f}) '
                        f'm=({mx:.1f},{my:.1f},{mz:.1f}) µT'
                    )

            # ---- Encoders -> Odometry ----
            elif 'enc' in js:
                arr = js['enc']
                if isinstance(arr, (list, tuple)) and len(arr) >= 4:
                    left_rot = float(arr[2])
                    right_rot = float(arr[3])

                    if self._first_run:
                        self._last_left_rot = left_rot
                        self._last_right_rot = right_rot
                        self._first_run = False

                    dl = left_rot - self._last_left_rot
                    dr = right_rot - self._last_right_rot
                    self._last_left_rot = left_rot
                    self._last_right_rot = right_rot

                    WHEEL_BASE = 0.074  # meters

                    d_center = (dl + dr) / 2.0
                    d_theta = (dr - dl) / WHEEL_BASE

                    self._theta += d_theta
                    self._x += d_center * math.cos(self._theta)
                    self._y += d_center * math.sin(self._theta)

                    msg = Odometry()
                    msg.header.stamp = self._now()
                    msg.header.frame_id = 'odom'
                    msg.child_frame_id = 'base_link'

                    msg.pose.pose.position.x = self._x
                    msg.pose.pose.position.y = self._y
                    msg.pose.pose.position.z = 0.0

                    # Yaw-only quaternion
                    half = self._theta * 0.5
                    qz = math.sin(half)
                    qw = math.cos(half)
                    msg.pose.pose.orientation.x = 0.0
                    msg.pose.pose.orientation.y = 0.0
                    msg.pose.pose.orientation.z = qz
                    msg.pose.pose.orientation.w = qw

                    # No velocity info here
                    msg.twist.twist.linear.x = 0.0
                    msg.twist.twist.linear.y = 0.0
                    msg.twist.twist.linear.z = 0.0
                    msg.twist.twist.angular.x = 0.0
                    msg.twist.twist.angular.y = 0.0
                    msg.twist.twist.angular.z = 0.0

                    msg.pose.covariance = [0.0] * 36
                    msg.twist.covariance = [-1.0] * 36

                    self.odom_raw_pub_.publish(msg)
                    self.get_logger().debug(
                        f'Odom: x={self._x:.3f} y={self._y:.3f} th={self._theta:.3f}'
                    )

            # ---- TOF -> Range ----
            elif 'tof' in js:
                arr = js['tof']
                if isinstance(arr, (list, tuple)) and len(arr) >= 3:
                    right_cm = float(arr[0])
                    left_cm = float(arr[1])
                    front_cm = float(arr[2])

                    def publish_range(dist_cm: float, pub, frame: str):
                        m = Range()
                        m.header.stamp = self._now()
                        m.header.frame_id = frame
                        m.radiation_type = Range.INFRARED
                        m.field_of_view = 0.1
                        m.min_range = 0.02
                        m.max_range = 2.0
                        if dist_cm <= 0.0 or dist_cm > 400.0:
                            m.range = math.nan
                        else:
                            m.range = dist_cm / 100.0
                        pub.publish(m)

                    publish_range(right_cm, self.tof_right_pub_, 'tof_right_link')
                    publish_range(left_cm, self.tof_left_pub_, 'tof_left_link')
                    publish_range(front_cm, self.tof_front_pub_, 'tof_front_link')

        except json.JSONDecodeError as e:
            # Provide byte/pos similar to C++ message
            self.get_logger().warn(f'JSON parse error at byte {e.pos}: {e}')
        except Exception as e:
            self.get_logger().warn(f'Error processing line: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
