#!/usr/bin/env python3
"""Lightweight RRT* planner + Pure Pursuit controller"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
import numpy as np
import cv2
from typing import List, Tuple, Optional
import time

class RRTNode:
    __slots__ = ['x', 'y', 'parent', 'cost']
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['RRTNode'] = None
        self.cost = 0.0
    
    def distance_to(self, other: 'RRTNode') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx*dx + dy*dy)

class MazeNavigator(Node):
    def __init__(self):
        super().__init__('maze_navigator')
        
        # Declare parameters
        self.declare_parameter('map.image_path', '/home/tron/maze_maps/maze.png')
        self.declare_parameter('map.real_width_m', 3.658)
        self.declare_parameter('map.real_height_m', 3.658)
        self.declare_parameter('map.robot_radius_m', 0.15)
        self.declare_parameter('planner.max_iterations', 1500)
        self.declare_parameter('planner.step_size_m', 0.08)
        self.declare_parameter('planner.goal_tolerance_m', 0.12)
        self.declare_parameter('controller.lookahead_distance_m', 0.25)
        self.declare_parameter('controller.max_linear_speed_ms', 0.4)
        self.declare_parameter('controller.max_angular_speed_rads', 1.5)
        
        # Get parameters
        self.map_image_path = self.get_parameter('map.image_path').value
        self.real_width = self.get_parameter('map.real_width_m').value
        self.real_height = self.get_parameter('map.real_height_m').value
        self.robot_radius = self.get_parameter('map.robot_radius_m').value
        self.max_iter = self.get_parameter('planner.max_iterations').value
        self.step_size = self.get_parameter('planner.step_size_m').value
        self.goal_tol = self.get_parameter('planner.goal_tolerance_m').value
        self.lookahead = self.get_parameter('controller.lookahead_distance_m').value
        self.max_linear = self.get_parameter('controller.max_linear_speed_ms').value
        self.max_angular = self.get_parameter('controller.max_angular_speed_rads').value
        
        # Load map
        self.load_map()
        
        # State
        self.current_pose: Optional[PoseStamped] = None
        self.current_path: Optional[List[Tuple[float, float]]] = None
        self.path_index = 0
        self.goal_reached = False
        
        # Publishers
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.create_subscription(Odometry, 'odom_filtered', self.odom_callback, 10)
        
        # Control timer
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Maze Navigator ready!')
    
    def load_map(self):
        """Load and process map image"""
        img = cv2.imread(self.map_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().warn(f'Cannot load map: {self.map_image_path}')
            # Create dummy map
            img = np.ones((500, 500), dtype=np.uint8) * 255
            self.get_logger().warn('Using dummy 500x500 map')
        
        self.img_height, self.img_width = img.shape
        self.resolution = self.real_width / self.img_width
        
        _, self.map_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Inflate obstacles
        kernel_size = int(self.robot_radius / self.resolution)
        if kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (kernel_size*2+1, kernel_size*2+1))
            self.map_inflated = cv2.erode(self.map_binary, kernel)
        else:
            self.map_inflated = self.map_binary
        
        self.get_logger().info(f'Map loaded: {self.img_width}x{self.img_height}, res={self.resolution*100:.2f}cm/px')
    
    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        px = int(x / self.resolution)
        py = int((self.real_height - y) / self.resolution)
        return px, py
    
    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        x = px * self.resolution
        y = self.real_height - (py * self.resolution)
        return x, y
    
    def is_free(self, point: Tuple[float, float]) -> bool:
        px, py = self.world_to_pixel(point[0], point[1])
        if px < 0 or px >= self.img_width or py < 0 or py >= self.img_height:
            return False
        return self.map_inflated[py, px] > 0
    
    def odom_callback(self, msg: Odometry):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.current_pose = pose
    
    def goal_callback(self, msg: PoseStamped):
        if self.current_pose is None:
            self.get_logger().warn('No current pose')
            return
        
        start = (self.current_pose.pose.position.x, 
                self.current_pose.pose.position.y)
        goal = (msg.pose.position.x, msg.pose.position.y)
        
        self.get_logger().info(f'Planning: ({start[0]:.2f},{start[1]:.2f})→({goal[0]:.2f},{goal[1]:.2f})')
        
        path = self.plan_rrt_star(start, goal)
        
        if path:
            self.current_path = path
            self.path_index = 0
            self.goal_reached = False
            self.publish_path(path)
            self.get_logger().info(f'Path found: {len(path)} waypoints')
        else:
            self.get_logger().error('No path found!')
            self.stop_robot()
    
    def plan_rrt_star(self, start: Tuple[float, float], 
                     goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Simplified RRT* for Pi Zero"""
        if not self.is_free(start) or not self.is_free(goal):
            return None
        
        start_node = RRTNode(start[0], start[1])
        goal_node = RRTNode(goal[0], goal[1])
        nodes = [start_node]
        best_goal = None
        
        for _ in range(self.max_iter):
            # Sample
            if np.random.random() < 0.15:
                rand = goal
            else:
                rand = (np.random.uniform(0, self.real_width),
                       np.random.uniform(0, self.real_height))
                if not self.is_free(rand):
                    continue
            
            # Nearest
            nearest = min(nodes, key=lambda n: np.hypot(n.x-rand[0], n.y-rand[1]))
            
            # Steer
            dx, dy = rand[0] - nearest.x, rand[1] - nearest.y
            dist = np.hypot(dx, dy)
            if dist < self.step_size:
                new_x, new_y = rand
            else:
                new_x = nearest.x + self.step_size * dx / dist
                new_y = nearest.y + self.step_size * dy / dist
            
            new_node = RRTNode(new_x, new_y)
            
            # Check collision (simple)
            if not self.is_free((new_x, new_y)):
                continue
            
            new_node.parent = nearest
            new_node.cost = nearest.cost + nearest.distance_to(new_node)
            nodes.append(new_node)
            
            # Check goal
            if new_node.distance_to(goal_node) < self.goal_tol:
                if best_goal is None or new_node.cost < best_goal.cost:
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + new_node.distance_to(goal_node)
                    best_goal = goal_node
        
        if best_goal:
            return self.extract_path(best_goal)
        return None
    
    def extract_path(self, goal: RRTNode) -> List[Tuple[float, float]]:
        path = []
        current = goal
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]
    
    def publish_path(self, waypoints: List[Tuple[float, float]]):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        
        self.path_pub.publish(msg)
    
    def control_loop(self):
        """Pure pursuit control"""
        if not self.current_path or not self.current_pose or self.goal_reached:
            return
        
        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        ryaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        
        # Find lookahead point
        goal_point = None
        for i in range(self.path_index, len(self.current_path)):
            px, py = self.current_path[i]
            dist = np.hypot(px - rx, py - ry)
            if dist >= self.lookahead:
                self.path_index = i
                goal_point = (px, py)
                break
        
        if goal_point is None:
            final_x, final_y = self.current_path[-1]
            if np.hypot(final_x - rx, final_y - ry) < 0.15:
                self.goal_reached = True
                self.stop_robot()
                self.get_logger().info('Goal reached!')
                return
            goal_point = (final_x, final_y)
        
        # Pure pursuit
        dx = goal_point[0] - rx
        dy = goal_point[1] - ry
        goal_angle = np.arctan2(dy, dx)
        angle_diff = goal_angle - ryaw
        while angle_diff > np.pi: angle_diff -= 2*np.pi
        while angle_diff < -np.pi: angle_diff += 2*np.pi
        
        linear = self.max_linear * np.cos(angle_diff)
        angular = 2.0 * linear * np.sin(angle_diff) / self.lookahead
        angular = np.clip(angular, -self.max_angular, self.max_angular)
        
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)
    
    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MazeNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
