import argparse
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # Optional dependency for serial control
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover
    serial = None

    class SerialException(Exception):
        """Fallback serial exception when pyserial is unavailable."""
        pass

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
        Returns: annotated_img, offset_px (float or None), alignment (str or None), largest_square tuple
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
            return annotated, offset_px, alignment, None
        
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
        
        return annotated, offset_px, alignment, largest_square


def _init_detector_from_args(args: argparse.Namespace) -> SquareDetection:
    params = {
        'gaussian_k': args.gaussian_k,
        'sigmaX': args.sigmaX,
        'lower_thresh': args.lower_thresh,
        'upper_thresh': args.upper_thresh,
        'min_area': args.min_area,
        'max_area': args.max_area,
        'square_tolerance': args.square_tolerance,
        'min_contour_length': args.min_contour_length,
        'target_size_mm': args.target_size_mm,
        'focal_length': args.focal_length,
    }
    return SquareDetection(params)


def _print_detection_summary(squares: Sequence[Tuple[int, int, int, int, int]], alignment: Optional[str], offset_px: Optional[float]) -> None:
    if not squares:
        print("No 40x40mm box detected.")
        return

    largest = max(squares, key=lambda s: s[4])
    x, y, w_obj, h_obj, area = largest
    print(f"Detected box at x={x}, y={y}, width={w_obj}px, height={h_obj}px, area={area}px^2")
    if alignment is not None and offset_px is not None:
        print(f"Image center offset: {offset_px:.1f}px -> alignment: {alignment}")


def _process_frame(frame: np.ndarray, detector: SquareDetection) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]], Optional[float], Optional[str], Optional[Tuple[int, int, int, int, int]]]:
    squares = detector.find_squares(frame)
    annotated, offset_px, alignment, largest_square = detector.draw_squares_and_center(frame, squares)
    return annotated, squares, offset_px, alignment, largest_square


def _handle_image(path: str, detector: SquareDetection, output_path: Optional[str], show_window: bool) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Unable to load image '{path}'")

    annotated, squares, offset_px, alignment, _ = _process_frame(frame, detector)
    _print_detection_summary(squares, alignment, offset_px)

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"Annotated image saved to {output_path}")

    if show_window:
        cv2.imshow("40mm Box Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _handle_video(source: int, detector: SquareDetection, output_path: Optional[str], show_window: bool, max_frames: Optional[int]) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {source}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, squares, offset_px, alignment, _ = _process_frame(frame, detector)
            _print_detection_summary(squares, alignment, offset_px)

            if writer:
                writer.write(annotated)

            if show_window:
                cv2.imshow("40mm Box Detection", annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()


class FeatureMatcher:
    def __init__(self, algorithm: str, reference_paths: Sequence[str], distance_threshold: float, match_threshold: int):
        if algorithm not in ("akaze", "orb"):
            raise ValueError(f"Unsupported feature detector '{algorithm}'")
        self.algorithm = algorithm
        self.distance_threshold = float(distance_threshold)
        self.match_threshold = int(match_threshold)

        if algorithm == "akaze":
            self.detector = cv2.AKAZE_create()
            self.norm_type = cv2.NORM_HAMMING
        else:
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.norm_type = cv2.NORM_HAMMING

        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=True)
        self.references = []

        for path in reference_paths:
            image = _load_image(path)
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
            if descriptors is None or len(keypoints) == 0:
                raise RuntimeError(f"{self.algorithm.upper()} could not compute descriptors for reference '{path}'")
            self.references.append({
                "name": os.path.basename(path),
                "keypoints": keypoints,
                "descriptors": descriptors
            })

        if not self.references:
            raise ValueError("No reference images were loaded for feature matching.")

    def match(self, frame: np.ndarray) -> dict:
        keypoints_frame, descriptors_frame = self.detector.detectAndCompute(frame, None)
        if descriptors_frame is None or len(keypoints_frame) == 0:
            return {
                "confirmed": False,
                "good_matches": 0,
                "total_matches": 0,
                "reference": None,
                "frame_keypoints": 0
            }

        best_result = {
            "confirmed": False,
            "good_matches": 0,
            "total_matches": 0,
            "reference": None,
            "frame_keypoints": len(keypoints_frame)
        }

        for ref in self.references:
            matches = self.matcher.match(descriptors_frame, ref["descriptors"])
            if not matches:
                continue
            good_matches = [m for m in matches if m.distance <= self.distance_threshold]
            good_count = len(good_matches)
            total_count = len(matches)

            if good_count > best_result["good_matches"]:
                best_result.update({
                    "good_matches": good_count,
                    "total_matches": total_count,
                    "reference": ref["name"]
                })

        if best_result["good_matches"] >= self.match_threshold:
            best_result["confirmed"] = True

        return best_result


class SerialTransmitter:
    def __init__(self, port: Optional[str], baudrate: int, dry_run: bool):
        self.port = port
        self.baudrate = baudrate
        self.dry_run = dry_run or not port
        self.handle = None

        if port and not self.dry_run:
            if serial is None:
                raise ImportError("pyserial is required for serial communication. Install it or use --dry-run.")
            try:
                self.handle = serial.Serial(port, baudrate, timeout=1)
                print(f"Opened serial port {port} at {baudrate} baud.")
            except SerialException as exc:  # pragma: no cover - hardware dependent
                raise RuntimeError(f"Failed to open serial port {port}: {exc}") from exc

    def send(self, command: str) -> None:
        message = command.strip()
        if not message:
            return

        print(f"[SERIAL] {message}")
        if self.handle:
            payload = (message + "\n").encode("ascii", errors="ignore")
            self.handle.write(payload)
            self.handle.flush()

    def close(self) -> None:
        if self.handle and self.handle.is_open:
            self.handle.close()


def _draw_roi(image: np.ndarray, min_width: float, max_width: float) -> None:
    h, w = image.shape[:2]
    center_x = w // 2
    center_y = h // 2
    roi_specs = (
        (max_width, (0, 255, 255)),
        (min_width, (0, 165, 255)),
    )
    for width, color in roi_specs:
        if width <= 0:
            continue
        half = int(width / 2)
        top_left = (max(center_x - half, 0), max(center_y - half, 0))
        bottom_right = (min(center_x + half, w - 1), min(center_y + half, h - 1))
        cv2.rectangle(image, top_left, bottom_right, color, 2)


def _draw_alignment_guides(image: np.ndarray, threshold: float) -> None:
    if threshold <= 0:
        return
    h, w = image.shape[:2]
    center_x = w // 2
    offset = int(round(threshold))
    for delta in (-offset, offset):
        x = center_x + delta
        x = max(0, min(w - 1, x))
        cv2.line(image, (x, 0), (x, h), (255, 255, 0), 1)


def _determine_command(match_confirmed: bool,
                       largest_square: Optional[Tuple[int, int, int, int, int]],
                       offset_px: Optional[float],
                       alignment_threshold: float,
                       roi_min_width: float,
                       roi_max_width: float,
                       commands: dict) -> str:
    if not match_confirmed or largest_square is None:
        return commands["search"]

    box_width = largest_square[2]

    if offset_px is not None:
        if offset_px < -alignment_threshold:
            return commands["left"]
        if offset_px > alignment_threshold:
            return commands["right"]

    if roi_min_width > 0 and box_width < roi_min_width:
        return commands["forward"]
    if roi_max_width > 0 and box_width > roi_max_width:
        return commands["backward"]
    return commands["stop"]


def _track_object(source: int,
                  detector: SquareDetection,
                  matcher: FeatureMatcher,
                  serial_tx: SerialTransmitter,
                  show_window: bool,
                  output_path: Optional[str],
                  max_frames: Optional[int],
                  alignment_threshold: float,
                  roi_min_width: float,
                  roi_max_width: float,
                  commands: dict) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {source}")

    writer = None
    if output_path:
        if os.path.isdir(output_path):
            video_path = os.path.join(output_path, "track_output.mp4")
        else:
            root, ext = os.path.splitext(output_path)
            video_path = output_path if ext else f"{output_path}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        print(f"Recording annotated tracking video to {video_path}")

    last_command = None
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            match_result = matcher.match(frame)
            annotated, squares, offset_px, alignment, largest_square = _process_frame(frame, detector)
            _draw_roi(annotated, roi_min_width, roi_max_width)
            _draw_alignment_guides(annotated, alignment_threshold)

            match_color = (0, 255, 0) if match_result["confirmed"] else (0, 0, 255)
            match_text = (f"{matcher.algorithm.upper()} matches: "
                          f"{match_result['good_matches']}/{match_result['total_matches']}")
            cv2.putText(annotated, match_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, match_color, 2)
            if match_result["reference"]:
                cv2.putText(annotated, f"Ref: {match_result['reference']}", (10, 50),
                            match_color, 0.6, 2)

            if largest_square is not None:
                cv2.putText(annotated, f"Box width: {largest_square[2]}px",
                            (10, 75), (255, 255, 255), 0.6, 2)

            command = _determine_command(
                match_confirmed=match_result["confirmed"],
                largest_square=largest_square,
                offset_px=offset_px,
                alignment_threshold=alignment_threshold,
                roi_min_width=roi_min_width,
                roi_max_width=roi_max_width,
                commands=commands,
            )

            if command != last_command:
                serial_tx.send(command)
                last_command = command

            cv2.putText(annotated, f"Command: {command}", (10, 100),
                        (255, 215, 0), 0.7, 2)

            if writer:
                writer.write(annotated)

            if show_window:
                cv2.imshow("40mm Box Tracking", annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image '{path}'")
    return image


def _resolve_output_path(base: Optional[str], algorithm: str) -> Optional[str]:
    if not base:
        return None
    if os.path.isdir(base):
        return os.path.join(base, f"{algorithm}_matches.png")
    root, ext = os.path.splitext(base)
    ext = ext or ".png"
    return f"{root}_{algorithm}{ext}"


def _run_feature_matching(algorithm: str, image_a_path: str, image_b_path: str,
                          output_path: Optional[str], show_window: bool,
                          distance_threshold: float, match_threshold: int) -> None:
    img_a = _load_image(image_a_path)
    img_b = _load_image(image_b_path)

    if algorithm == "akaze":
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING
    else:
        detector = cv2.ORB_create(nfeatures=1000)
        norm_type = cv2.NORM_HAMMING

    kp_a, des_a = detector.detectAndCompute(img_a, None)
    kp_b, des_b = detector.detectAndCompute(img_b, None)

    if des_a is None or des_b is None:
        raise RuntimeError(f"{algorithm.upper()} could not find descriptors in one of the images.")

    matcher = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = matcher.match(des_a, des_b)
    matches = sorted(matches, key=lambda m: m.distance)

    good_matches = [m for m in matches if m.distance <= distance_threshold]
    confirmation = "CONFIRMED" if len(good_matches) >= match_threshold else "INSUFFICIENT"

    matched_viz = cv2.drawMatches(
        img_a, kp_a, img_b, kp_b, matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print(f"{algorithm.upper()} found {len(kp_a)} keypoints in {image_a_path} "
          f"and {len(kp_b)} in {image_b_path}; {len(good_matches)} good matches "
          f"(distance ≤ {distance_threshold}) -> {confirmation}.")

    if output_path:
        resolved = _resolve_output_path(output_path, algorithm)
        cv2.imwrite(resolved, matched_viz)
        print(f"Match visualization saved to {resolved}")

    if show_window:
        cv2.imshow(f"{algorithm.upper()} Matches", matched_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a 40x40mm target, perform feature matching, or track the cube with motion commands."
    )
    parser.add_argument("--mode", choices=["square", "akaze", "orb", "both", "track"], default="square",
                        help="Processing mode: square detection, standalone feature matching, or live tracking.")
    parser.add_argument("--source", default="0",
                        help="Path to an image/video file or camera index (square/track modes; default: 0).")
    parser.add_argument("--output", help="Path (file or directory) to save annotated output.")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames (video sources).")
    parser.add_argument("--no-window", action="store_true", help="Disable preview window.")
    parser.add_argument("--image-a", default="cube_1.png", help="First reference image for feature matching modes.")
    parser.add_argument("--image-b", default="cube_2.png", help="Second reference image for feature matching modes.")
    parser.add_argument("--feature-detector", choices=["akaze", "orb"], default="orb",
                        help="Feature detector to use for tracking/matching (track mode default: orb).")
    parser.add_argument("--match-threshold", type=int, default=25,
                        help="Minimum number of good matches to confirm the reference cube.")
    parser.add_argument("--match-distance", type=float, default=60.0,
                        help="Maximum descriptor distance to count as a good match.")
    parser.add_argument("--serial-port", help="Serial port device path for motion commands (e.g. /dev/ttyUSB0).")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate (default: 115200).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print motion commands instead of opening the serial port.")
    parser.add_argument("--cmd-forward", default="FORWARD", help="Command string to move forward.")
    parser.add_argument("--cmd-backward", default="BACKWARD", help="Command string to move backward.")
    parser.add_argument("--cmd-left", default="LEFT", help="Command string to strafe/turn left.")
    parser.add_argument("--cmd-right", default="RIGHT", help="Command string to strafe/turn right.")
    parser.add_argument("--cmd-stop", default="STOP", help="Command string when the cube is aligned within ROI.")
    parser.add_argument("--cmd-search", default="SEARCH",
                        help="Command string when the cube is not detected; typically rotate in place.")
    parser.add_argument("--alignment-threshold", type=float, default=20.0,
                        help="Acceptable horizontal pixel offset from image center before commanding left/right.")
    parser.add_argument("--roi-min-width", type=float, default=80.0,
                        help="Minimum bounding-box width (pixels) before commanding FORWARD.")
    parser.add_argument("--roi-max-width", type=float, default=140.0,
                        help="Maximum bounding-box width (pixels) before commanding BACKWARD.")
    parser.add_argument("--gaussian-k", nargs=2, type=int, default=[5, 5], metavar=("KX", "KY"))
    parser.add_argument("--sigmaX", type=float, default=1.0)
    parser.add_argument("--lower-thresh", type=int, default=50)
    parser.add_argument("--upper-thresh", type=int, default=150)
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--max-area", type=int, default=3000)
    parser.add_argument("--square-tolerance", type=float, default=0.2)
    parser.add_argument("--min-contour-length", type=int, default=40)
    parser.add_argument("--target-size-mm", type=float, default=40.0)
    parser.add_argument("--focal-length", type=float, default=500.0)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    show_window = not args.no_window
    mode = args.mode

    if args.roi_max_width > 0 and args.roi_min_width > 0 and args.roi_max_width < args.roi_min_width:
        raise ValueError("roi-max-width must be greater than or equal to roi-min-width when both are positive.")

    if mode == "square":
        detector = _init_detector_from_args(args)
        source_str = args.source
        source_is_number = source_str.isdigit() or (source_str.startswith("-") and source_str[1:].isdigit())

        if not source_is_number and not os.path.exists(source_str):
            raise FileNotFoundError(f"Source '{source_str}' does not exist")

        if not source_is_number and os.path.isfile(source_str):
            _handle_image(source_str, detector, args.output, show_window)
        else:
            video_source = int(source_str) if source_is_number else source_str
            _handle_video(video_source, detector, args.output, show_window, args.max_frames)
        return

    if mode in ("akaze", "orb", "both"):
        algorithms = ("akaze", "orb") if mode == "both" else (mode,)
        for algorithm in algorithms:
            _run_feature_matching(
                algorithm=algorithm,
                image_a_path=args.image_a,
                image_b_path=args.image_b,
                output_path=args.output,
                show_window=show_window,
                distance_threshold=args.match_distance,
                match_threshold=args.match_threshold,
            )
        return

    if mode == "track":
        detector = _init_detector_from_args(args)
        reference_paths = [args.image_a, args.image_b]
        matcher = FeatureMatcher(
            algorithm=args.feature_detector,
            reference_paths=reference_paths,
            distance_threshold=args.match_distance,
            match_threshold=args.match_threshold,
        )
        commands = {
            "forward": args.cmd_forward,
            "backward": args.cmd_backward,
            "left": args.cmd_left,
            "right": args.cmd_right,
            "stop": args.cmd_stop,
            "search": args.cmd_search,
        }
        serial_tx = SerialTransmitter(args.serial_port, args.baudrate, args.dry_run)
        try:
            source_str = args.source
            source_is_number = source_str.isdigit() or (source_str.startswith("-") and source_str[1:].isdigit())
            if not source_is_number and not os.path.exists(source_str):
                raise FileNotFoundError(f"Source '{source_str}' does not exist")
            video_source = int(source_str) if source_is_number else source_str
            _track_object(
                source=video_source,
                detector=detector,
                matcher=matcher,
                serial_tx=serial_tx,
                show_window=show_window,
                output_path=args.output,
                max_frames=args.max_frames,
                alignment_threshold=args.alignment_threshold,
                roi_min_width=args.roi_min_width,
                roi_max_width=args.roi_max_width,
                commands=commands,
            )
        finally:
            serial_tx.close()
        return

    raise ValueError(f"Unsupported mode '{mode}'")


if __name__ == '__main__':
    main()
