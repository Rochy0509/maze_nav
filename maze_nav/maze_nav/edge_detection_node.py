import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class EdgeDetection(Node):

    def __init__(self):
        super().__init__("edge_detection_node")
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Declare parameters
        self.declare_parameter('gaussian_k', [5, 5])
        self.declare_parameter('sigmaX', 1.0)
        self.declare_parameter('lower_thresh', 50)
        self.declare_parameter('upper_thresh', 150)
        self.declare_parameter('hough_threshold', 30)
        self.declare_parameter('min_line_length', 50)
        self.declare_parameter('max_line_gap', 30)
        self.declare_parameter('wall_bottom_region', 0.7)

        # Get parameters
        self.gaussian_k = self.get_parameter('gaussian_k').value
        self.sigmaX = self.get_parameter('sigmaX').value
        self.lower_thresh = self.get_parameter('lower_thresh').value
        self.upper_thresh = self.get_parameter('upper_thresh').value
        self.hough_threshold = self.get_parameter('hough_threshold').value
        self.min_line_length = self.get_parameter('min_line_length').value
        self.max_line_gap = self.get_parameter('max_line_gap').value
        self.wall_bottom_region = self.get_parameter('wall_bottom_region').value

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        
        # Publisher for annotated image
        self.publisher = self.create_publisher(Image, '/edge_image_annotated', 10)
        
        self.get_logger().info('Edge Detection Node Started')

    def listener_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect edges
            blurred, canny_edges = self.detect_edges(cv_img)
            
            # Draw lines and detect gaps
            annotated = self.draw_lines(cv_img, canny_edges)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self.publisher.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_edges(self, img):
        """Detect edges using Canny"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(src=gray, ksize=tuple(self.gaussian_k), sigmaX=self.sigmaX)
        
        # Apply Canny edge detection
        canny_edges = cv2.Canny(blurred, self.lower_thresh, self.upper_thresh)

        return blurred, canny_edges
    
    def get_line_equation(self, x1, y1, x2, y2):
        """Get slope and intercept of a line"""
        if x2 - x1 == 0:
            return None, x1  # Vertical line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    
    def get_x_at_y(self, y, slope, intercept):
        """Get x coordinate at given y for a line"""
        if slope == 0:
            return None
        return (y - intercept) / slope
    
    def draw_dashed_line(self, img, pt1, pt2, color, thickness=2, dash_length=20):
        """Draw a dashed line"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        # Unit vector
        ux = dx / length
        uy = dy / length
        
        # Draw dashes
        current_length = 0
        while current_length < length:
            # Start of dash
            start_x = int(x1 + ux * current_length)
            start_y = int(y1 + uy * current_length)
            
            # End of dash
            end_length = min(current_length + dash_length, length)
            end_x = int(x1 + ux * end_length)
            end_y = int(y1 + uy * end_length)
            
            # Draw dash
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            
            # Move to next dash (skip gap)
            current_length += dash_length * 2
    
    def draw_lines(self, img, edges):
        """
        Detect lines from edges for perspective view maze.
        Identifies left and right wall edges (blue) and center path (red dashed)
        """
        # Create a copy for annotation
        annotated = img.copy()
        height, width = edges.shape
        
        # Define wall region (lower portion of image for perspective)
        wall_region_start = int(height * (1 - self.wall_bottom_region))
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            self.get_logger().warn('No lines detected')
            return annotated
        
        self.get_logger().info(f'Detected {len(lines)} total lines')
        
        # Separate lines into left wall, right wall
        left_wall_lines = []
        right_wall_lines = []
        center_x = width / 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle of line (in degrees)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Calculate average y position (prioritize lines in lower region)
            avg_y = (y1 + y2) / 2
            
            if avg_y > wall_region_start:  # In the wall region
                avg_x = (x1 + x2) / 2
                line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Left wall lines (negative slope, left side)
                if avg_x < center_x and -80 < angle < -10:
                    left_wall_lines.append((x1, y1, x2, y2, angle, line_length))
                    cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue
                
                # Right wall lines (positive slope, right side)
                elif avg_x > center_x and 10 < angle < 80:
                    right_wall_lines.append((x1, y1, x2, y2, angle, line_length))
                    cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue
        
        self.get_logger().info(f'Left wall lines: {len(left_wall_lines)}, Right wall lines: {len(right_wall_lines)}')
        
        # Draw yellow reference line (image center) in wall region only
        image_center = width // 2
        cv2.line(annotated, (image_center, wall_region_start), (image_center, height-1), (0, 255, 255), 3)
        
        # Calculate and draw center path
        if left_wall_lines and right_wall_lines:
            # Find the longest/best lines for each wall
            left_best = max(left_wall_lines, key=lambda l: l[5])  # longest line
            right_best = max(right_wall_lines, key=lambda l: l[5])
            
            # Get line equations
            left_slope, left_intercept = self.get_line_equation(
                left_best[0], left_best[1], left_best[2], left_best[3]
            )
            right_slope, right_intercept = self.get_line_equation(
                right_best[0], right_best[1], right_best[2], right_best[3]
            )
            
            # Calculate center points at various y coordinates
            center_points = []
            y_step = 20  # Sample every 20 pixels
            
            for y in range(wall_region_start, height, y_step):
                # Get x coordinates on both walls at this y
                left_x = self.get_x_at_y(y, left_slope, left_intercept)
                right_x = self.get_x_at_y(y, right_slope, right_intercept)
                
                if left_x is not None and right_x is not None:
                    # Calculate midpoint
                    center_x_at_y = (left_x + right_x) / 2
                    center_points.append((int(center_x_at_y), y))
            
            # Also add points from top to the first wall point
            if center_points:
                # Extrapolate upward
                if len(center_points) >= 2:
                    # Use the top two points to extrapolate
                    p1 = center_points[0]
                    p2 = center_points[1]
                    
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    if dy != 0:
                        # Extrapolate to top of image
                        steps = (p1[1] - 0) // y_step
                        for i in range(1, int(steps) + 1):
                            new_y = p1[1] - i * y_step
                            if new_y >= 0:
                                new_x = p1[0] - (dx / dy) * (i * y_step)
                                center_points.insert(0, (int(new_x), int(new_y)))
            
            # Draw dashed center line through all points (RED)
            if len(center_points) >= 2:
                for i in range(len(center_points) - 1):
                    self.draw_dashed_line(
                        annotated,
                        center_points[i],
                        center_points[i + 1],
                        (0, 0, 255),  # Red
                        thickness=3,
                        dash_length=15
                    )
                
                # Calculate offset at bottom
                bottom_center = center_points[-1][0]
                offset = bottom_center - image_center
                
                self.get_logger().info(f'Center offset at bottom: {offset:.2f} pixels')
                
                # Add alignment text
                alignment = "LEFT" if offset < -20 else "RIGHT" if offset > 20 else "CENTER"
                cv2.putText(annotated, f"Align: {alignment}", (width//2 - 100, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                cv2.putText(annotated, f"Offset: {offset:.0f}px", (width//2 - 100, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Detect gaps/openings for turning
        if len(left_wall_lines) < 2:
            cv2.line(annotated, (10, height-50), (10, height-10), (0, 255, 0), 8)
            cv2.putText(annotated, "TURN LEFT", (20, height-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            self.get_logger().info('LEFT TURN DETECTED')
        
        if len(right_wall_lines) < 2:
            cv2.line(annotated, (width-10, height-50), (width-10, height-10), (0, 255, 0), 8)
            cv2.putText(annotated, "TURN RIGHT", (width-200, height-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            self.get_logger().info('RIGHT TURN DETECTED')
        
        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = EdgeDetection()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()