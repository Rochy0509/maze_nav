import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range, Imu
from nav_msgs.msg import Odometry
import time
import math

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        
        # QoS for sensor data
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # State machine states
        self.state = 'FIND_GAP'  # Start by finding the first gap
        self.turn_count = 0
        self.moving_forward = False
        self.target_angle = 0.0
        self.start_time = 0.0
        self.forward_duration = 0.0
        self.turn_start_time = 0.0
        
        # TOF sensor data
        self.front_dist = float('inf')
        self.right_dist = float('inf')
        self.left_dist = float('inf')
        
        # Edge detection data
        self.offset_px = 0.0
        self.alignment = "CENTER"
        
        # IMU data
        self.current_yaw = 0.0
        self.initial_yaw = 0.0
        
        # Odometry data for return to start
        self.current_x = 0.0
        self.current_y = 0.0
        self.start_pose = None
        self.pose_history = []
        
        # Dead-end detection and escape
        self.dead_end_detected = False
        self.escape_start_time = 0.0
        self.escape_duration = 2.0  # Time to reverse in dead end
        self.escape_distance = 0.3  # Distance to reverse (meters)
        self.initial_escape_yaw = 0.0  # Yaw when escape started
        self.escape_state = 'NONE'  # NONE, DETECTED, ESCAPING, COMPLETED
        
        # Navigation history for path tracking
        self.nav_history = []  # Store navigation decisions
        self.max_history_length = 20  # Limit history size
        
        # Robot parameters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.5  # rad/s
        self.reverse_speed = -0.15  # m/s (negative for reverse)
        self.min_gap_distance = 0.3  # meters
        self.center_tolerance = 20  # pixels for alignment
        self.wall_distance_setpoint = 0.08  # Target distance from wall (80mm)
        self.wall_follow_kp = 1.0  # Proportional gain for wall following
        
        # Robot dimensions
        self.robot_width = 0.119  # 119mm
        self.min_corridor_width = 0.150  # 150mm
        
        # Task completion tracking
        self.package_found = False
        self.task_completed = False
        self.return_to_start_after_task = True  # Automatically return after finding package
        
        # Return to start parameters
        self.returning = False
        self.return_path = []
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)
        
        # Subscribers
        self.sub_tof_0 = self.create_subscription(Range, '/tof/sensor_0', self.tof_0_callback, qos)
        self.sub_tof_1 = self.create_subscription(Range, '/tof/sensor_1', self.tof_1_callback, qos)
        self.sub_tof_2 = self.create_subscription(Range, '/tof/sensor_2', self.tof_2_callback, qos)
        
        # Edge detection
        self.sub_offset = self.create_subscription(Float32, '/edge_detection/offset_px', self.offset_callback, qos)
        self.sub_alignment = self.create_subscription(String, '/edge_detection/alignment', self.alignment_callback, qos)
        
        # IMU for precise turning
        self.sub_imu = self.create_subscription(Imu, '/imu/data', self.imu_callback, qos)
        
        # Odometry for position tracking
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        
        # Timer for navigation loop
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)  # 10Hz
        
        # Timer for status publishing
        self.status_timer = self.create_timer(1.0, self.publish_status)  # 1Hz status updates
        
        self.get_logger().info("Autonomous Navigation Node initialized")

    def tof_0_callback(self, msg):
        self.front_dist = msg.range

    def tof_1_callback(self, msg):
        self.right_dist = msg.range

    def tof_2_callback(self, msg):
        self.left_dist = msg.range

    def offset_callback(self, msg):
        self.offset_px = msg.data

    def alignment_callback(self, msg):
        self.alignment = msg.data

    def imu_callback(self, msg):
        orientation = msg.orientation
        siny_cosp = 2 * (msg.orientation.w * msg.orientation.z + msg.orientation.x * msg.orientation.y)
        cosy_cosp = 1 - 2 * (msg.orientation.y * msg.orientation.y + msg.orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_yaw = yaw

    def odom_callback(self, msg):
        # Store current position
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Store initial pose if not set
        if self.start_pose is None:
            self.start_pose = (self.current_x, self.current_y, self.current_yaw)
        
        # Store pose history for potential return path
        if not self.returning and not self.task_completed:
            self.pose_history.append((self.current_x, self.current_y, self.current_yaw))

    def check_dead_end(self):
        """Check if robot is in a dead end based on TOF sensor readings."""
        # Dead end: front, left, and right are all blocked (or very close to obstacles)
        front_blocked = self.front_dist < 0.4  # 40cm threshold
        right_blocked = self.right_dist < 0.3  # 30cm threshold
        left_blocked = self.left_dist < 0.3    # 30cm threshold
        
        # If all three directions are blocked, we're in a dead end
        if front_blocked and right_blocked and left_blocked:
            return True
        
        # Also check for very narrow passages that might be dead ends
        corridor_width = self.right_dist + self.left_dist
        if front_blocked and corridor_width < 0.2:  # Less than 20cm total width
            return True
            
        return False

    def get_yaw_difference(self, target_yaw, current_yaw):
        """Calculate the shortest angular difference between two angles."""
        diff = target_yaw - current_yaw
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def calculate_wall_follow_correction(self):
        """Calculate lateral correction based on TOF sensors for wall following."""
        correction = 0.0
        
        # Use right and left TOF sensors to stay centered
        if self.right_dist != float('inf') and self.left_dist != float('inf'):
            # Calculate desired distance from each wall (centered)
            desired_right = (self.right_dist + self.left_dist) / 2.0
            desired_left = desired_right
            
            # Calculate error from desired center position
            right_error = self.right_dist - desired_right
            left_error = self.left_dist - desired_left
            
            # Use the more reliable side (whichever is closer to the wall)
            if self.right_dist < self.left_dist:
                # Following right wall
                error = self.right_dist - self.wall_distance_setpoint
                correction = self.wall_follow_kp * error
            else:
                # Following left wall
                error = self.wall_distance_setpoint - self.left_dist
                correction = -self.wall_follow_kp * error
        elif self.right_dist != float('inf'):
            # Only right sensor available - follow right wall
            error = self.right_dist - self.wall_distance_setpoint
            correction = self.wall_follow_kp * error
        elif self.left_dist != float('inf'):
            # Only left sensor available - follow left wall
            error = self.wall_distance_setpoint - self.left_dist
            correction = -self.wall_follow_kp * error
        
        # Limit the correction to prevent excessive steering
        correction = max(-0.5, min(0.5, correction))
        return correction

    def calculate_edge_center_correction(self):
        """Calculate correction based on edge detection alignment."""
        correction = 0.0
        
        if self.alignment == "LEFT":
            correction = -0.1  # Turn right
        elif self.alignment == "RIGHT":
            correction = 0.1   # Turn left
        # If CENTER, no correction needed
        
        return correction

    def publish_status(self):
        """Publish current navigation status."""
        status_msg = String()
        if self.task_completed:
            if self.returning:
                status_msg.data = f"TASK_COMPLETED_RETURNING: {self.state}"
            else:
                status_msg.data = "TASK_COMPLETED_AT_START"
        elif self.package_found:
            status_msg.data = f"PACKAGE_FOUND_STATE: {self.state}"
        else:
            status_msg.data = f"NAVIGATING: {self.state} | Turns: {self.turn_count} | Escape: {self.escape_state}"
        self.status_pub.publish(status_msg)

    def navigation_loop(self):
        cmd_vel = Twist()
        
        # Current sensor readings
        front_dist = self.front_dist
        right_dist = self.right_dist
        left_dist = self.left_dist
        
        # Check for dead end
        self.dead_end_detected = self.check_dead_end()
        
        # Handle dead end escape if detected
        if self.dead_end_detected and self.escape_state == 'NONE':
            self.escape_state = 'DETECTED'
            self.get_logger().info("DEAD END DETECTED - INITIATING ESCAPE")
        
        # Handle escape behavior
        if self.escape_state == 'DETECTED':
            self.escape_state = 'ESCAPING'
            self.escape_start_time = time.time()
            self.initial_escape_yaw = self.current_yaw
        elif self.escape_state == 'ESCAPING':
            # Reverse for a set time
            elapsed = time.time() - self.escape_start_time
            
            if elapsed < self.escape_duration:
                # Reverse while maintaining original heading
                cmd_vel.linear.x = self.reverse_speed  # Negative for reverse
                # Maintain original heading during reverse
                yaw_diff = self.get_yaw_difference(self.initial_escape_yaw, self.current_yaw)
                cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, yaw_diff * 2))
                
                self.get_logger().info(f"ESCAPING DEAD END: {elapsed:.1f}s of {self.escape_duration}s")
            else:
                # Done reversing
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.escape_state = 'COMPLETED'
                self.get_logger().info("ESCAPE COMPLETED - RESUMING NAVIGATION")
                
                # Add a small random turn to try different path after escape
                self.initial_yaw = self.current_yaw + math.radians(45)  # 45 degree offset
        elif self.escape_state == 'COMPLETED':
            # Reset escape state and resume navigation
            self.escape_state = 'NONE'
            # Continue with navigation logic below

        # Main navigation logic (only if not escaping)
        if self.escape_state != 'ESCAPING':
            if self.returning:
                # Handle return to start
                if self.state == 'RETURN_TO_START':
                    # Calculate direction to start position
                    dx = self.start_pose[0] - self.current_x
                    dy = self.start_pose[1] - self.current_y
                    distance_to_start = math.sqrt(dx*dx + dy*dy)
                    
                    if distance_to_start > 0.1:  # 10cm tolerance
                        # Calculate target angle to start position
                        target_angle = math.atan2(dy, dx)
                        angle_diff = self.get_yaw_difference(target_angle, self.current_yaw)
                        
                        # Move toward start
                        cmd_vel.linear.x = min(self.linear_speed, distance_to_start * 0.5)  # Slower as we get closer
                        cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 2))
                        
                        self.get_logger().info(f"RETURNING TO START: {distance_to_start:.2f}m away")
                    else:
                        # Reached start position
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0
                        self.task_completed = True
                        self.get_logger().info("RETURNED TO START - TASK COMPLETE!")
            elif self.task_completed:
                # Task completed, stay still
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.get_logger().info("TASK COMPLETED - STAYING AT START")
            elif self.package_found:
                # Package found, decide whether to return to start
                if self.return_to_start_after_task:
                    self.returning = True
                    self.state = 'RETURN_TO_START'
                    self.get_logger().info("PACKAGE FOUND - RETURNING TO START")
                else:
                    # Task completed without return
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.0
                    self.task_completed = True
                    self.get_logger().info("PACKAGE FOUND - TASK COMPLETED")
            else:
                # Normal navigation
                if self.state == 'FIND_GAP':
                    # Move forward while maintaining center position
                    if right_dist > self.min_gap_distance and front_dist > 0.5:
                        cmd_vel.linear.x = self.linear_speed
                        # Apply corrections for centering
                        wall_correction = self.calculate_wall_follow_correction()
                        edge_correction = self.calculate_edge_center_correction()
                        cmd_vel.angular.z = wall_correction + edge_correction * 0.5
                        self.get_logger().info("MOVING FORWARD TO FIND GAP WITH CENTERING")
                    elif right_dist <= self.min_gap_distance:
                        # Gap found, prepare to turn right
                        self.state = 'TURN_RIGHT'
                        self.initial_yaw = self.current_yaw
                        self.turn_start_time = time.time()
                        self.get_logger().info("GAP DETECTED ON RIGHT, PREPARING TO TURN")
                    else:
                        # Obstacle ahead, need to handle
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0
                        self.get_logger().info("OBSTACLE AHEAD WHILE FINDING GAP")

                elif self.state == 'TURN_RIGHT':
                    # Turn right using IMU feedback for precise 90-degree turn
                    target_yaw = self.initial_yaw - math.pi/2  # 90 degrees right
                    yaw_diff = self.get_yaw_difference(target_yaw, self.current_yaw)
                    
                    if abs(yaw_diff) > 0.1:  # 0.1 rad = ~5.7 degrees tolerance
                        cmd_vel.angular.z = -min(self.angular_speed, max(-self.angular_speed, yaw_diff * 2))
                        cmd_vel.linear.x = 0.0
                        self.get_logger().info(f"TURNING RIGHT: {math.degrees(yaw_diff):.1f}° remaining")
                    else:
                        self.turn_count += 1
                        self.get_logger().info(f"TURNED RIGHT. TURN COUNT: {self.turn_count}")
                        
                        if self.turn_count == 3:
                            # After 3rd turn, look for gap on right again
                            self.state = 'FIND_GAP_RIGHT'
                        else:
                            # Continue to next segment
                            self.state = 'MOVE_FORWARD'
                            self.start_time = time.time()
                            self.forward_duration = 3.0
                        self.get_logger().info(f"NEW STATE: {self.state}")

                elif self.state == 'MOVE_FORWARD':
                    # Move forward for a set duration while centering
                    elapsed = time.time() - self.start_time
                    if elapsed < self.forward_duration:
                        cmd_vel.linear.x = self.linear_speed
                        wall_correction = self.calculate_wall_follow_correction()
                        edge_correction = self.calculate_edge_center_correction()
                        cmd_vel.angular.z = wall_correction + edge_correction * 0.5
                        self.get_logger().info("MOVING FORWARD IN CORRIDOR WITH CENTERING")
                    else:
                        # Duration complete, look for next decision point
                        self.state = 'FIND_GAP'

                elif self.state == 'FIND_GAP_RIGHT':
                    # After 3rd turn, look for gap in right wall
                    if right_dist > self.min_gap_distance and front_dist > 0.5:
                        cmd_vel.linear.x = self.linear_speed
                        wall_correction = self.calculate_wall_follow_correction()
                        edge_correction = self.calculate_edge_center_correction()
                        cmd_vel.angular.z = wall_correction + edge_correction * 0.5
                        self.get_logger().info("MOVING FORWARD TOWARD RIGHT GAP WITH CENTERING")
                    elif right_dist <= self.min_gap_distance:
                        # Found gap on right, turn right again
                        self.state = 'TURN_RIGHT_GAP'
                        self.initial_yaw = self.current_yaw
                        self.turn_start_time = time.time()
                        self.get_logger().info("GAP FOUND IN RIGHT WALL, TURNING RIGHT AGAIN")
                    else:
                        # Obstacle ahead
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0

                elif self.state == 'TURN_RIGHT_GAP':
                    # Turn right into the gap using IMU feedback
                    target_yaw = self.initial_yaw - math.pi/2  # 90 degrees right
                    yaw_diff = self.get_yaw_difference(target_yaw, self.current_yaw)
                    
                    if abs(yaw_diff) > 0.1:
                        cmd_vel.angular.z = -min(self.angular_speed, max(-self.angular_speed, yaw_diff * 2))
                        cmd_vel.linear.x = 0.0
                    else:
                        self.state = 'FIND_GAP_LEFT'
                        self.get_logger().info("TURNED INTO RIGHT GAP, NOW LOOKING FOR LEFT GAP")

                elif self.state == 'FIND_GAP_LEFT':
                    # Look for gap on left side of maze
                    if left_dist > self.min_gap_distance and front_dist > 0.5:
                        cmd_vel.linear.x = self.linear_speed
                        wall_correction = self.calculate_wall_follow_correction()
                        edge_correction = self.calculate_edge_center_correction()
                        cmd_vel.angular.z = wall_correction + edge_correction * 0.5
                        self.get_logger().info("MOVING FORWARD IN MAZE WITH CENTERING, LOOKING FOR LEFT GAP")
                    elif left_dist <= self.min_gap_distance:
                        # Found gap on left, turn left to find package
                        self.state = 'TURN_LEFT_PACKAGE'
                        self.initial_yaw = self.current_yaw
                        self.turn_start_time = time.time()
                        self.get_logger().info("GAP FOUND ON LEFT, TURNING LEFT FOR PACKAGE")
                    else:
                        # Obstacle ahead
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0

                elif self.state == 'TURN_LEFT_PACKAGE':
                    # Turn left to go toward package using IMU feedback
                    target_yaw = self.initial_yaw + math.pi/2  # 90 degrees left
                    yaw_diff = self.get_yaw_difference(target_yaw, self.current_yaw)
                    
                    if abs(yaw_diff) > 0.1:
                        cmd_vel.angular.z = min(self.angular_speed, max(-self.angular_speed, yaw_diff * 2))
                        cmd_vel.linear.x = 0.0
                    else:
                        self.package_found = True
                        self.get_logger().info("PACKAGE FOUND - TASK COMPLETION INITIATED!")

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

def main():
    rclpy.init()
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()