import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge

class SquareDetection:
    def __init__(self, params: dict):
        # Tunables for square detection
        self.gaussian_k = params.get('gaussian_k', [5, 5])
        self.sigmaX = float(params.get('sigmaX', 1.0))
        self.lower_thresh = int(params.get('lower_thresh', 50))
        self.upper_thresh = int(params.get('upper_thresh', 150))
        self.min_area = int(params.get('min_area', 800))  # Minimum area for 40x40mm box
        self.max_area = int(params.get('max_area', 3000))  # Maximum area for 40x40mm box
        self.square_tolerance = float(params.get('square_tolerance', 0.2))  # How close to 1:1 ratio
        self.min_contour_length = int(params.get('min_contour_length', 40))  # Minimum contour length
        self.target_size_mm = float(params.get('target_size_mm', 40.0))  # Target size in mm
        self.focal_length = float(params.get('focal_length', 500.0))  # Camera focal length in pixels

    def detect_edges(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=tuple(self.gaussian_k), sigmaX=self.sigmaX)
        canny_edges = cv2.Canny(blurred, self.lower_thresh, self.upper_thresh)
        return blurred, canny_edges

    def find_squares(self, img_bgr):
        """
        Find square objects in the image.
        Returns: list of (x, y, w, h, area) for detected squares
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=tuple(self.gaussian_k), sigmaX=self.sigmaX)
        edges = cv2.Canny(blurred, self.lower_thresh, self.upper_thresh)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        squares = []
        for contour in contours:
            # Filter by contour length
            if cv2.arcLength(contour, True) < self.min_contour_length:
                continue
                
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a quadrilateral
            if len(approx) == 4:
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                
                # Check if it's within size range
                if self.min_area <= area <= self.max_area:
                    # Check if it's approximately square
                    aspect_ratio = float(w) / h
                    if 1.0 - self.square_tolerance <= aspect_ratio <= 1.0 + self.square_tolerance:
                        squares.append((x, y, w, h, area))
        
        return squares

    def get_distance_to_object(self, width_in_pixels):
        """
        Calculate distance to object based on known size and focal length.
        """
        # Distance = (known_width * focal_length) / pixel_width
        distance = (self.target_size_mm * self.focal_length) / width_in_pixels
        return distance  # in same units as target_size_mm

    def draw_squares_and_center(self, img_bgr, squares):
        """
        Draw detected squares and calculate center offset.
        Returns: annotated_img, offset_px (float or None), alignment (str or None)
        """
        annotated = img_bgr.copy()
        h, w = img_bgr.shape[:2]
        
        if not squares:
            # No squares detected
            offset_px = None
            alignment = "NONE"
            # Draw "NO BOX" indicator
            cv2.putText(annotated, "NO BOX DETECTED", (w//2 - 100, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return annotated, offset_px, alignment
        
        # Find the largest square (closest/focused object)
        largest_square = max(squares, key=lambda s: s[4])  # Sort by area
        x, y, w_obj, h_obj, area = largest_square
        
        # Draw the detected square
        cv2.rectangle(annotated, (x, y), (x + w_obj, y + h_obj), (0, 255, 0), 3)
        cv2.putText(annotated, f"40x40mm Box", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate object center
        obj_center_x = x + w_obj // 2
        obj_center_y = y + h_obj // 2
        
        # Draw center of object
        cv2.circle(annotated, (obj_center_x, obj_center_y), 5, (0, 0, 255), -1)
        
        # Calculate offset from image center
        center_x_img = w // 2
        offset_px = float(obj_center_x - center_x_img)
        alignment = "LEFT" if offset_px < -20 else "RIGHT" if offset_px > 20 else "CENTER"
        
        # Draw center line
        cv2.line(annotated, (center_x_img, 0), (center_x_img, h), (255, 255, 0), 2)
        
        # Draw offset indicator
        cv2.putText(annotated, f"Align: {alignment}", (w//2 - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(annotated, f"Offset: {int(offset_px)}px", (w//2 - 120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw distance estimation
        distance = self.get_distance_to_object(w_obj)
        cv2.putText(annotated, f"Dist: {distance:.1f}mm", (w//2 - 120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        return annotated, offset_px, alignment


class SquareDetectionNode(Node):
    def __init__(self):
        super().__init__('square_detection_node')

        # QoS tuned for camera streams
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Declare & read params
        self.declare_parameter('gaussian_k', [5, 5])
        self.declare_parameter('sigmaX', 1.0)
        self.declare_parameter('lower_thresh', 50)
        self.declare_parameter('upper_thresh', 150)
        self.declare_parameter('min_area', 800)
        self.declare_parameter('max_area', 3000)
        self.declare_parameter('square_tolerance', 0.2)
        self.declare_parameter('min_contour_length', 40)
        self.declare_parameter('target_size_mm', 40.0)
        self.declare_parameter('focal_length', 500.0)  # Calibrate this value
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('publish_compressed', False)

        self.publish_compressed = bool(self.get_parameter('publish_compressed').get_parameter_value().bool_value)
        params = {p: self.get_parameter(p).value for p in [
            'gaussian_k','sigmaX','lower_thresh','upper_thresh','min_area',
            'max_area','square_tolerance','min_contour_length','target_size_mm','focal_length'
        ]}
        self.proc = SquareDetection(params)

        self.bridge = CvBridge()
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        # Publishers
        self.pub_annotated = self.create_publisher(Image, '/square_detection/annotated', qos)
        self.pub_squares   = self.create_publisher(Image, '/square_detection/edges', qos)
        self.pub_offset    = self.create_publisher(Float32, '/square_detection/offset_px', 10)
        self.pub_align     = self.create_publisher(String,  '/square_detection/alignment', 10)
        self.pub_annotated_compressed = None
        if self.publish_compressed:
            self.pub_annotated_compressed = self.create_publisher(CompressedImage, '/square_detection/annotated/compressed', 10)

        # Subscriber
        self.sub = self.create_subscription(Image, input_topic, self.image_cb, qos)

        # Allow live param updates
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(f"Subscribed to: {input_topic}")
        self.get_logger().info("SquareDetectionNode ready.")

    def _on_param_update(self, params):
        changed = []
        for p in params:
            name = p.name
            if hasattr(self.proc, name):
                setattr(self.proc, name, p.value)
                changed.append((name, p.value))
            elif name == 'publish_compressed':
                self.publish_compressed = bool(p.value)
                if self.publish_compressed and self.pub_annotated_compressed is None:
                    self.pub_annotated_compressed = self.create_publisher(CompressedImage, '/square_detection/annotated/compressed', 10)
                if not self.publish_compressed and self.pub_annotated_compressed is not None:
                    # No direct "destroy publisher" in rclpy; just stop using it.
                    self.pub_annotated_compressed = None
                changed.append((name, self.publish_compressed))
        if changed:
            self.get_logger().info(f"Updated params: {changed}")
        return rclpy.parameter.SetParametersResult(successful=True)

    def _from_nv21_to_bgr(self, msg: Image):
        """Manual NV21 (YUV420sp) -> BGR conversion (avoids cv_bridge encoding error)."""
        w, h = msg.width, msg.height
        yuv = np.frombuffer(msg.data, dtype=np.uint8)
        expected = int(h * 1.5) * w
        if yuv.size != expected:
            self.get_logger().warn(f"NV21 size mismatch: got {yuv.size}, expected {expected}")
            return None
        yuv = yuv.reshape((int(h * 1.5), w))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return bgr

    def image_cb(self, msg: Image):
        # Convert ROS Image -> OpenCV BGR
        try:
            enc = (msg.encoding or '').lower()
            if enc in ('bgr8', 'rgb8', 'mono8'):
                # Let cv_bridge handle common encodings
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif enc in ('nv21', 'yuv420sp'):
                bgr = self._from_nv21_to_bgr(msg)
                if bgr is None:
                    return
            else:
                # Fallback: ask cv_bridge to coerce to bgr
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Run square detection pipeline
        squares = self.proc.find_squares(bgr)
        annotated, offset_px, alignment = self.proc.draw_squares_and_center(bgr, squares)

        # Publish images (keep original header stamp/frame_id)
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.pub_annotated.publish(annotated_msg)

            # Publish edges image (showing detected contours)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
            edges_msg.header = msg.header
            self.pub_squares.publish(edges_msg)
        except Exception as e:
            self.get_logger().error(f"Publishing image failed: {e}")

        # Optional compressed annotated
        if self.publish_compressed and self.pub_annotated_compressed is not None:
            try:
                ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    cmsg = CompressedImage()
                    cmsg.header = msg.header
                    cmsg.format = 'jpeg'
                    cmsg.data = buf.tobytes()
                    self.pub_annotated_compressed.publish(cmsg)
            except Exception as e:
                self.get_logger().warn(f"Compressed publish failed: {e}")

        # Publish numeric outputs
        if offset_px is not None:
            self.pub_offset.publish(Float32(data=float(offset_px)))
        if alignment is not None:
            self.pub_align.publish(String(data=alignment))


def main():
    rclpy.init()
    node = SquareDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()