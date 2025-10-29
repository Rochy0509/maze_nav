import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge

class EdgeDetection:
    def __init__(self, params: dict):
        # Tunables (can be updated on-the-fly via ROS params callback)
        self.gaussian_k = params.get('gaussian_k', [5, 5])
        self.sigmaX = float(params.get('sigmaX', 1.0))
        self.lower_thresh = int(params.get('lower_thresh', 30))
        self.upper_thresh = int(params.get('upper_thresh', 135))
        self.hough_threshold = int(params.get('hough_threshold', 20))
        self.min_line_length = int(params.get('min_line_length', 60))
        self.max_line_gap = int(params.get('max_line_gap', 35))
        self.wall_bottom_region = float(params.get('wall_bottom_region', 0.5))

    def detect_edges(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=tuple(self.gaussian_k), sigmaX=self.sigmaX)
        canny_edges = cv2.Canny(blurred, self.lower_thresh, self.upper_thresh)
        return blurred, canny_edges

    @staticmethod
    def get_line_equation(x1, y1, x2, y2):
        if x2 - x1 == 0:
            return None, x1  # Vertical: slope None, use x-intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    @staticmethod
    def get_x_at_y(y, slope, intercept):
        if slope is None or slope == 0:
            return None
        return (y - intercept) / slope

    @staticmethod
    def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=20):
        x1, y1 = pt1
        x2, y2 = pt2
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        cur = 0.0
        while cur < length:
            sx = int(x1 + ux * cur); sy = int(y1 + uy * cur)
            e = min(cur + dash_length, length)
            ex = int(x1 + ux * e); ey = int(y1 + uy * e)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
            cur += dash_length * 2

    def draw_lines_and_center(self, img_bgr, edges):
        """
        Returns: annotated_img, offset_px (float or None), alignment (str or None)
        """
        annotated = img_bgr.copy()
        h, w = edges.shape
        wall_region_start = int(h * (1.0 - self.wall_bottom_region))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        left_wall_lines = []
        right_wall_lines = []
        center_x_img = w / 2.0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                avg_y = 0.5 * (y1 + y2)
                if avg_y > wall_region_start:
                    avg_x = 0.5 * (x1 + x2)
                    if avg_x < center_x_img and -80 < angle < -10:
                        left_wall_lines.append((x1, y1, x2, y2, angle, float(np.hypot(x2-x1, y2-y1))))
                        cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    elif avg_x > center_x_img and 10 < angle < 80:
                        right_wall_lines.append((x1, y1, x2, y2, angle, float(np.hypot(x2-x1, y2-y1))))
                        cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Yellow image-center line in wall region
        cv2.line(annotated, (int(w/2), wall_region_start), (int(w/2), h-1), (0, 255, 255), 3)

        offset_px = None
        alignment = None

        if left_wall_lines and right_wall_lines:
            left_best  = max(left_wall_lines, key=lambda t: t[5])
            right_best = max(right_wall_lines, key=lambda t: t[5])

            lm, lb = self.get_line_equation(left_best[0], left_best[1], left_best[2], left_best[3])
            rm, rb = self.get_line_equation(right_best[0], right_best[1], right_best[2], right_best[3])

            center_pts = []
            y_step = 20
            for y in range(wall_region_start, h, y_step):
                lx = self.get_x_at_y(y, lm, lb)
                rx = self.get_x_at_y(y, rm, rb)
                if lx is not None and rx is not None:
                    cx = 0.5 * (lx + rx)
                    center_pts.append((int(cx), int(y)))

            # Extrapolate upwards for nicer dashed line
            if len(center_pts) >= 2:
                p1, p2 = center_pts[0], center_pts[1]
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                if dy != 0:
                    steps = max(0, (p1[1] - 0) // y_step)
                    for i in range(1, int(steps)+1):
                        ny = p1[1] - i*y_step
                        if ny >= 0:
                            nx = p1[0] - (dx/dy) * (i*y_step)
                            center_pts.insert(0, (int(nx), int(ny)))

            # Draw dashed centerline (red) and compute bottom offset
            if len(center_pts) >= 2:
                for i in range(len(center_pts)-1):
                    self.draw_dashed_line(annotated, center_pts[i], center_pts[i+1], (0, 0, 255), thickness=3, dash_length=15)

                bottom_center_x = center_pts[-1][0]
                offset_px = float(bottom_center_x - (w/2.0))
                alignment = "LEFT" if offset_px < -20 else "RIGHT" if offset_px > 20 else "CENTER"

                cv2.putText(annotated, f"Align: {alignment}", (w//2 - 120, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(annotated, f"Offset: {int(offset_px)}px", (w//2 - 120, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Simple gap/turn hints
        if len(left_wall_lines) < 2:
            cv2.line(annotated, (10, h-50), (10, h-10), (0, 255, 0), 8)
            cv2.putText(annotated, "TURN LEFT", (20, h-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        if len(right_wall_lines) < 2:
            cv2.line(annotated, (w-10, h-50), (w-10, h-10), (0, 255, 0), 8)
            cv2.putText(annotated, "TURN RIGHT", (w-200, h-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        return annotated, offset_px, alignment


class EdgeDetectionNode(Node):
    def __init__(self):
        super().__init__('edge_detection_node')

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
        self.declare_parameter('hough_threshold', 30)
        self.declare_parameter('min_line_length', 50)
        self.declare_parameter('max_line_gap', 30)
        self.declare_parameter('wall_bottom_region', 0.7)
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('publish_compressed', False)

        self.publish_compressed = bool(self.get_parameter('publish_compressed').get_parameter_value().bool_value)
        params = {p: self.get_parameter(p).value for p in [
            'gaussian_k','sigmaX','lower_thresh','upper_thresh','hough_threshold',
            'min_line_length','max_line_gap','wall_bottom_region'
        ]}
        self.proc = EdgeDetection(params)

        self.bridge = CvBridge()
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        # Publishers
        self.pub_annotated = self.create_publisher(Image, '/edge_detection/annotated', qos)
        self.pub_edges     = self.create_publisher(Image, '/edge_detection/edges', qos)
        self.pub_offset    = self.create_publisher(Float32, '/edge_detection/offset_px', 10)
        self.pub_align     = self.create_publisher(String,  '/edge_detection/alignment', 10)
        self.pub_annotated_compressed = None
        if self.publish_compressed:
            self.pub_annotated_compressed = self.create_publisher(CompressedImage, '/edge_detection/annotated/compressed', 10)

        # Subscriber
        self.sub = self.create_subscription(Image, input_topic, self.image_cb, qos)

        # Allow live param updates
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(f"Subscribed to: {input_topic}")
        self.get_logger().info("EdgeDetectionNode ready.")

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
                    self.pub_annotated_compressed = self.create_publisher(CompressedImage, '/edge_detection/annotated/compressed', 10)
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

        # Run pipeline
        _, edges = self.proc.detect_edges(bgr)
        annotated, offset_px, alignment = self.proc.draw_lines_and_center(bgr, edges)

        # Publish images (keep original header stamp/frame_id)
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.pub_annotated.publish(annotated_msg)

            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
            edges_msg.header = msg.header
            self.pub_edges.publish(edges_msg)
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
    node = EdgeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()