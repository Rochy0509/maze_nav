#!/usr/bin/env python3
"""Simple goal sender for testing"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import sys

class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')
        self.pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
    
    def send_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 1.0
        
        self.pub.publish(msg)
        self.get_logger().info(f'Goal sent: ({x}, {y})')

def main(args=None):
    rclpy.init(args=args)
    
    if len(sys.argv) < 3:
        print('Usage: ros2 run maze_nav send_goal <x> <y>')
        sys.exit(1)
    
    node = GoalSender()
    node.send_goal(sys.argv[1], sys.argv[2])
    
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
