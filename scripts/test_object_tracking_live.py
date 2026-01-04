#!/usr/bin/env python3
"""Test live object tracking from IP Webcam feed.

This script:
1. Connects to IP Webcam stream from phone
2. Takes text prompts to find objects
3. Tracks objects using one of three methods:
   - Follow Anything (FAn) with SAM 2
   - Follow Anything Fallback (Grounding DINO + Bot-SORT)
   - Grounding DINO + Bot-SORT (direct)

Test objects: "desk lamp" and "scissors"
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
import os

import cv2
import numpy as np
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.appconfig import AppConfig
from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    GroundingDINOModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import (
    FollowAnythingFallback,
    FollowAnythingModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
    BotSORTTracker,
)
from ahmedkobtan_cinema_robot_playground.src.services.video_stream import VideoStream
from ahmedkobtan_cinema_robot_playground.src.utils.config_utils import parse_config


class ObjectTracker:
    """Object tracker using different methods."""

    def __init__(self, method: str = "fan", device: str = "cpu"):
        """
        Initialize tracker.

        Args:
            method: Tracking method ('fan', 'fan_fallback', 'grounding_dino')
            device: Device to use ('cuda' or 'cpu')
        """
        self.method = method
        self.device = device
        self.model = None
        self.tracker = None
        self.fallback = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models based on method."""
        logger.info(f"Initializing {self.method} tracking method...")

        if self.method == "fan":
            # Follow Anything with SAM 2
            import os

            # Set SAM2_CHECKPOINT if checkpoint exists in resources
            if not os.getenv("SAM2_CHECKPOINT"):
                resources_dir = (
                    project_root / "ahmedkobtan_cinema_robot_playground" / "resources"
                )
                default_checkpoints = [
                    resources_dir / "sam2.1_hiera_base_plus.pt",
                    resources_dir / "sam2.1_hiera_large.pt",
                    resources_dir / "sam2.1_hiera_small.pt",
                    resources_dir / "sam2.1_hiera_tiny.pt",
                ]

                for checkpoint_path in default_checkpoints:
                    if checkpoint_path.exists():
                        os.environ["SAM2_CHECKPOINT"] = str(checkpoint_path)
                        logger.info(f"Using SAM 2 checkpoint: {checkpoint_path.name}")
                        break

            self.model = FollowAnythingModel(use_sam2=True, device=self.device)
            if not self.model.is_available() and self.model.fallback is None:
                logger.warning("FAn not available, falling back to FAn fallback")
                self.method = "fan_fallback"
                self._initialize_models()
                return

        elif self.method == "fan_fallback":
            # Follow Anything Fallback (Grounding DINO + Bot-SORT)
            self.fallback = FollowAnythingFallback(device=self.device)

        elif self.method == "grounding_dino":
            # Grounding DINO + Bot-SORT directly
            self.model = GroundingDINOModel(device=self.device)
            self.tracker = BotSORTTracker(device=self.device)
            if not self.tracker.is_available():
                logger.warning("Bot-SORT not available, tracking may fail")

        logger.info(f"✓ {self.method} method initialized")

    def detect_and_track(
        self, frame: np.ndarray, text_prompt: str, initial_detection: bool = True
    ):
        """
        Detect and track object in frame.

        Args:
            frame: Video frame (BGR format)
            text_prompt: Text description of object
            initial_detection: Whether this is initial detection (True) or tracking update (False)

        Returns:
            Tuple of (success, bounding_box, tracking_state)
        """
        try:
            if self.method == "fan":
                if initial_detection:
                    # Initial detection
                    boxes = self.model.detect(frame, text_prompt)
                    if boxes and len(boxes) > 0:
                        bbox = boxes[0]
                        h, w = frame.shape[:2]
                        # Reject full-frame bounding boxes using area coverage (more robust)
                        bbox_area = bbox.width * bbox.height
                        image_area = w * h
                        area_coverage = bbox_area / image_area if image_area > 0 else 0
                        is_full_frame = area_coverage >= 0.95  # Cover >95% of image
                        # Only use if confidence is reasonable and not full frame
                        # Lower threshold to 0.25 to match FAn's internal threshold
                        if bbox.confidence > 0.25 and not is_full_frame:
                            initial_bbox = (bbox.x, bbox.y, bbox.width, bbox.height)
                            state = self.model.update(frame, initial_bbox=initial_bbox)
                            return (True, bbox, state)
                        elif is_full_frame:
                            logger.warning(
                                f"Rejected full-frame detection (conf={bbox.confidence:.2f})"
                            )
                    return (False, None, None)
                else:
                    # Continue tracking
                    state = self.model.update(frame)
                    if state:
                        x, y, w, h = state.bbox
                        from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
                            BoundingBox,
                        )

                        bbox = BoundingBox(
                            x=x, y=y, width=w, height=h, confidence=state.confidence
                        )
                        return (True, bbox, state)
                    return (False, None, None)

            elif self.method == "fan_fallback":
                if initial_detection:
                    boxes = self.fallback.detect(frame, text_prompt)
                    if boxes and len(boxes) > 0:
                        # Sort by confidence and try best matches
                        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
                        for bbox in boxes:
                            h, w = frame.shape[:2]
                            # Reject full-frame bounding boxes using area coverage
                            bbox_area = bbox.width * bbox.height
                            image_area = w * h
                            area_coverage = (
                                bbox_area / image_area if image_area > 0 else 0
                            )
                            is_full_frame = area_coverage >= 0.95
                            # Use reasonable confidence threshold (Grounding DINO uses 0.3 internally)
                            if bbox.confidence > 0.25 and not is_full_frame:
                                initial_bbox = (bbox.x, bbox.y, bbox.width, bbox.height)
                                state = self.fallback.update(
                                    frame, initial_bbox=initial_bbox
                                )
                                return (True, bbox, state)
                            elif is_full_frame:
                                logger.debug(
                                    f"Skipping full-frame detection (conf={bbox.confidence:.2f})"
                                )
                    return (False, None, None)
                else:
                    state = self.fallback.update(frame)
                    if state:
                        x, y, w, h = state.bbox
                        from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
                            BoundingBox,
                        )

                        bbox = BoundingBox(
                            x=x, y=y, width=w, height=h, confidence=state.confidence
                        )
                        return (True, bbox, state)
                    return (False, None, None)

            elif self.method == "grounding_dino":
                if initial_detection:
                    # Initial detection with Grounding DINO
                    boxes = self.model.detect(frame, text_prompt)
                    if boxes and len(boxes) > 0:
                        # Sort by confidence and try best matches
                        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
                        for bbox in boxes:
                            h, w = frame.shape[:2]
                            # Reject full-frame bounding boxes using area coverage
                            bbox_area = bbox.width * bbox.height
                            image_area = w * h
                            area_coverage = (
                                bbox_area / image_area if image_area > 0 else 0
                            )
                            is_full_frame = area_coverage >= 0.95
                            # Use reasonable confidence threshold (Grounding DINO uses 0.3 internally)
                            if bbox.confidence > 0.25 and not is_full_frame:
                                if self.tracker.is_available():
                                    initial_bbox = (
                                        bbox.x,
                                        bbox.y,
                                        bbox.width,
                                        bbox.height,
                                    )
                                    state = self.tracker.update(
                                        frame, initial_bbox=initial_bbox
                                    )
                                    return (True, bbox, state)
                                else:
                                    # Tracker not available, return detection only
                                    return (True, bbox, None)
                            elif is_full_frame:
                                logger.debug(
                                    f"Skipping full-frame detection (conf={bbox.confidence:.2f})"
                                )
                    return (False, None, None)
                else:
                    # Continue tracking
                    if self.tracker.is_available():
                        state = self.tracker.update(frame)
                        if state:
                            x, y, w, h = state.bbox
                            from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
                                BoundingBox,
                            )

                            bbox = BoundingBox(
                                x=x, y=y, width=w, height=h, confidence=state.confidence
                            )
                            return (True, bbox, state)
                    return (False, None, None)

        except Exception as e:
            logger.error(f"Error in detect_and_track: {e}")
            return (False, None, None)


def draw_tracking_box(frame: np.ndarray, bbox, label: str = "Object"):
    """Draw bounding box on frame."""
    if bbox is None:
        return frame

    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw label
    label_text = f"{label} ({bbox.confidence:.2f})"
    (text_width, text_height), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1
    )
    cv2.putText(
        frame,
        label_text,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )

    return frame


def create_test_image_with_objects() -> np.ndarray:
    """Create a test image with simple objects for testing."""
    # Create a 1280x720 image
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Add some colored rectangles to simulate objects
    # "Desk lamp" - yellow rectangle on left
    cv2.rectangle(img, (200, 300), (400, 500), (0, 255, 255), -1)
    cv2.putText(img, "LAMP", (250, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # "Scissors" - blue rectangle on right
    cv2.rectangle(img, (800, 200), (1000, 350), (255, 0, 0), -1)
    cv2.putText(
        img, "SCISSORS", (810, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    # Add some noise to make it more realistic
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


def test_tracking(
    stream_url: str,
    text_prompt: str,
    method: str = "fan",
    device: str = "cpu",
    max_frames: int = 300,
    test_mode: bool = False,
):
    """
    Test object tracking from IP Webcam stream.

    Args:
        stream_url: IP Webcam URL (ignored if test_mode=True)
        text_prompt: Text description of object to track
        method: Tracking method ('fan', 'fan_fallback', 'grounding_dino')
        device: Device to use ('cuda' or 'cpu')
        max_frames: Maximum frames to process
        test_mode: If True, use test images instead of live stream
    """
    logger.info("=" * 60)
    logger.info("Live Object Tracking Test")
    logger.info("=" * 60)
    if test_mode:
        logger.info("TEST MODE: Using generated test images")
    else:
        logger.info(f"Stream URL: {stream_url}")
    logger.info(f"Text Prompt: {text_prompt}")
    logger.info(f"Method: {method}")
    logger.info(f"Device: {device}")
    logger.info("")

    # Initialize tracker
    tracker = ObjectTracker(method=method, device=device)

    # Connect to video stream or use test mode
    if test_mode:
        stream = None
        logger.info("✓ Test mode enabled - using generated frames")
    else:
        # Use URL exactly as provided (no automatic /video append)
        # User can provide full URL including endpoint if needed
        stream = VideoStream(stream_url=stream_url, width=1280, height=720, fps=30)
        logger.info(f"Attempting to connect to: {stream_url}")
        if not stream.connect():
            logger.error("Failed to connect to video stream")
            logger.info("\nTroubleshooting:")
            logger.info("1. Make sure IP Webcam app is running on your phone")
            logger.info("2. Check that phone and PC are on the same Wi-Fi network")
            logger.info("3. Verify the IP address and port in the URL")
            logger.info(
                "4. Some IP Webcam apps may require /video suffix (e.g., http://IP:PORT/video)"
            )
            logger.info("5. Check IP Webcam app settings for the correct stream URL")
            logger.info("\nTip: Use --test-mode to test without live stream")
            return False

    if not test_mode:
        logger.info("✓ Connected to video stream")
        logger.info("Press 'q' to quit, 'r' to re-detect, 's' to save frame")

    # Tracking state
    tracking_active = False
    frame_count = 0
    successful_tracks = 0
    failed_tracks = 0
    start_time = time.time()

    try:
        while frame_count < max_frames:
            if test_mode:
                # Generate test frame
                frame = create_test_image_with_objects()
                success = True
            else:
                success, frame = stream.read_frame()
                if not success or frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue

            frame_count += 1
            display_frame = frame.copy()

            # Detect or track
            # Re-detect every 10 frames to refresh detection (Bot-SORT needs periodic detections)
            REDETECT_INTERVAL = 10

            if not tracking_active:
                # Initial detection
                logger.info(f"Frame {frame_count}: Detecting '{text_prompt}'...")
                success, bbox, state = tracker.detect_and_track(
                    frame, text_prompt, initial_detection=True
                )

                if success and bbox:
                    tracking_active = True
                    logger.info(
                        f"✓ Object detected: x={bbox.x:.0f}, y={bbox.y:.0f}, "
                        f"w={bbox.width:.0f}, h={bbox.height:.0f}, conf={bbox.confidence:.2f}"
                    )
                    display_frame = draw_tracking_box(display_frame, bbox, text_prompt)
                    successful_tracks += 1
                else:
                    logger.warning("✗ Object not detected")
                    failed_tracks += 1
                    cv2.putText(
                        display_frame,
                        f"Searching for: {text_prompt}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
            else:
                # Continue tracking - re-detect periodically to refresh
                # Bot-SORT can maintain tracks for a few frames using Kalman filter,
                # but we need to re-detect periodically for accuracy
                if frame_count % REDETECT_INTERVAL == 0:
                    # Periodic re-detection to refresh the track
                    logger.debug(
                        f"Frame {frame_count}: Re-detecting '{text_prompt}'..."
                    )
                    success, bbox, state = tracker.detect_and_track(
                        frame, text_prompt, initial_detection=True
                    )
                    if success and bbox:
                        # Re-detection successful, track refreshed
                        logger.debug(
                            f"✓ Track refreshed: x={bbox.x:.0f}, y={bbox.y:.0f}, "
                            f"w={bbox.width:.0f}, h={bbox.height:.0f}, conf={bbox.confidence:.2f}"
                        )
                    else:
                        # Re-detection failed, try to continue with existing track
                        success, bbox, state = tracker.detect_and_track(
                            frame, text_prompt, initial_detection=False
                        )
                else:
                    # Continue tracking without re-detection
                    # Bot-SORT will use Kalman filter prediction
                    success, bbox, state = tracker.detect_and_track(
                        frame, text_prompt, initial_detection=False
                    )

                if success and bbox:
                    display_frame = draw_tracking_box(display_frame, bbox, text_prompt)
                    successful_tracks += 1

                    # Show tracking info
                    if state:
                        cv2.putText(
                            display_frame,
                            f"Track ID: {state.track_id}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                else:
                    logger.warning(f"Frame {frame_count}: Tracking lost")
                    failed_tracks += 1
                    tracking_active = False
                    cv2.putText(
                        display_frame,
                        "Tracking lost - Re-detecting...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # Calculate and display FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f} | Frame: {frame_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # Display frame (skip in test mode if no display available)
            if not test_mode:
                cv2.imshow("Object Tracking", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    logger.info("Re-detecting object...")
                    tracking_active = False
                elif key == ord("s"):
                    filename = f"tracked_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_frame)
                    logger.info(f"Saved frame to: {filename}")
            else:
                # In test mode, just log progress
                if frame_count % 10 == 0:
                    logger.info(
                        f"Frame {frame_count}: Tracking active={tracking_active}, "
                        f"Success={successful_tracks}, Failed={failed_tracks}"
                    )
                time.sleep(0.033)  # Simulate ~30 FPS

    except KeyboardInterrupt:
        logger.info("\nStopped by user")

    finally:
        if not test_mode:
            cv2.destroyAllWindows()
            if stream:
                stream.disconnect()

    # Print summary
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    success_rate = (
        successful_tracks / (successful_tracks + failed_tracks) * 100
        if (successful_tracks + failed_tracks) > 0
        else 0
    )

    logger.info("\n" + "=" * 60)
    logger.info("Tracking Test Results")
    logger.info("=" * 60)
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Successful tracks: {successful_tracks}")
    logger.info(f"Failed tracks: {failed_tracks}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    return success_rate > 50  # Consider success if >50% tracking rate


def main():
    """Run object tracking tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test live object tracking from IP Webcam"
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default=None,
        help="IP Webcam URL (default: from config file). "
        "Note: Some IP Webcam apps may require /video suffix, adjust as needed.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for object to track (e.g., 'desk lamp', 'scissors')",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fan", "fan_fallback", "grounding_dino"],
        default="fan",
        help="Tracking method: 'fan' (Follow Anything), 'fan_fallback' (FAn fallback), "
        "'grounding_dino' (Grounding DINO + Bot-SORT)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda', default: cpu)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum frames to process (default: 300)",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all methods with 'desk lamp' and 'scissors'",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test images instead of live stream (for testing without phone)",
    )

    args = parser.parse_args()

    # Get default URL from config if not provided
    if args.stream_url is None:
        try:
            # Load config inline (no helper function)
            env = os.getenv("ENV", "dev").lower()
            project_root = Path(__file__).parent.parent
            config_file = project_root / "config" / f"{env}.json"
            if not config_file.exists():
                config_file = project_root / "config" / "dev.json"
            config_data = parse_config(str(config_file))
            app_config = AppConfig(**config_data)
            args.stream_url = app_config.configResolution.resolved.ip_webcam_url
            logger.info(f"Using IP webcam URL from config: {args.stream_url}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Please provide --stream-url explicitly")
            return 1

    if args.test_all:
        # Test all methods with both objects
        test_objects = ["desk lamp", "scissors"]
        methods = ["fan", "fan_fallback", "grounding_dino"]

        results = {}
        for method in methods:
            results[method] = {}
            for obj in test_objects:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Testing: {method} with '{obj}'")
                logger.info(f"{'=' * 60}\n")
                success = test_tracking(
                    args.stream_url,
                    obj,
                    method=method,
                    device=args.device,
                    max_frames=args.max_frames,
                    test_mode=args.test_mode,
                )
                results[method][obj] = success
                time.sleep(2)  # Brief pause between tests

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("All Tests Summary")
        logger.info("=" * 60)
        for method in methods:
            logger.info(f"\n{method.upper()}:")
            for obj in test_objects:
                status = "✓ PASS" if results[method][obj] else "✗ FAIL"
                logger.info(f"  {obj}: {status}")

        all_passed = all(
            results[method][obj] for method in methods for obj in test_objects
        )
        return 0 if all_passed else 1
    else:
        # Single test
        success = test_tracking(
            args.stream_url,
            args.prompt,
            method=args.method,
            device=args.device,
            max_frames=args.max_frames,
            test_mode=args.test_mode,
        )
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
