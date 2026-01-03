#!/usr/bin/env python3
"""Test video streaming from phone to PC."""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
import cv2
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.appconfig import AppConfig
from ahmedkobtan_cinema_robot_playground.src.services.video_stream import VideoStream
from ahmedkobtan_cinema_robot_playground.src.utils.config_utils import parse_config


def test_local_camera():
    """Test local camera (USB webcam)."""
    logger.info("=" * 60)
    logger.info("Testing Local Camera")
    logger.info("=" * 60)

    try:
        stream = VideoStream(camera_index=0, width=1280, height=720, fps=30)
        if not stream.connect():
            logger.error("Failed to connect to local camera")
            return False

        logger.info("✓ Connected to local camera")
        logger.info(f"Frame size: {stream.get_frame_size()}")

        # Read a few frames
        for i in range(10):
            success, frame = stream.read_frame()
            if success and frame is not None:
                logger.info(f"  Frame {i + 1}: {frame.shape}")
            else:
                logger.warning(f"  Frame {i + 1}: Failed to read")
            time.sleep(0.1)

        stream.disconnect()
        logger.info("✓ Local camera test completed")
        return True

    except Exception as e:
        logger.error(f"Error testing local camera: {e}")
        return False


def test_ip_webcam(stream_url: str):
    """
    Test IP Webcam stream.

    Args:
        stream_url: URL for IP Webcam (e.g., "http://192.168.1.100:8080/video")
    """
    logger.info("=" * 60)
    logger.info("Testing IP Webcam Stream")
    logger.info("=" * 60)
    logger.info(f"Stream URL: {stream_url}")

    try:
        stream = VideoStream(stream_url=stream_url, width=1280, height=720, fps=30)
        if not stream.connect():
            logger.error("Failed to connect to IP Webcam stream")
            logger.info("\nTroubleshooting:")
            logger.info("1. Make sure IP Webcam app is running on your phone")
            logger.info("2. Check that phone and PC are on the same Wi-Fi network")
            logger.info("3. Verify the IP address and port in the URL")
            logger.info("4. Try opening the URL in a web browser first")
            return False

        logger.info("✓ Connected to IP Webcam stream")
        width, height = stream.get_frame_size()
        logger.info(f"Frame size: {width}x{height}")

        # Read frames and measure performance
        frame_count = 0
        start_time = time.time()
        successful_frames = 0
        failed_frames = 0

        logger.info("\nReading frames (press Ctrl+C to stop)...")
        try:
            for i in range(100):  # Read 100 frames
                success, frame = stream.read_frame()
                if success and frame is not None:
                    successful_frames += 1
                    frame_count += 1
                    if i % 10 == 0:
                        logger.info(
                            f"  Frame {i + 1}: {frame.shape}, Success rate: {successful_frames / (i + 1) * 100:.1f}%"
                        )
                else:
                    failed_frames += 1
                    logger.warning(f"  Frame {i + 1}: Failed to read")

                time.sleep(0.033)  # ~30 FPS

        except KeyboardInterrupt:
            logger.info("\nStopped by user")

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        logger.info("\n" + "=" * 60)
        logger.info("Stream Test Results")
        logger.info("=" * 60)
        logger.info(f"Total frames read: {frame_count}")
        logger.info(f"Successful frames: {successful_frames}")
        logger.info(f"Failed frames: {failed_frames}")
        logger.info(
            f"Success rate: {successful_frames / (successful_frames + failed_frames) * 100:.1f}%"
        )
        logger.info(f"Average FPS: {fps:.2f}")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        stream.disconnect()
        logger.info("✓ IP Webcam test completed")
        return True

    except Exception as e:
        logger.error(f"Error testing IP Webcam: {e}")
        return False


def test_video_stream_with_display(stream_url: str = None, camera_index: int = 0):
    """
    Test video stream with OpenCV display window.

    Args:
        stream_url: URL for IP Webcam (None for local camera)
        camera_index: Local camera index (if stream_url is None)
    """
    logger.info("=" * 60)
    logger.info("Testing Video Stream with Display")
    logger.info("=" * 60)

    try:
        if stream_url:
            stream = VideoStream(stream_url=stream_url, width=1280, height=720, fps=30)
            logger.info(f"Connecting to: {stream_url}")
        else:
            stream = VideoStream(
                camera_index=camera_index, width=1280, height=720, fps=30
            )
            logger.info(f"Connecting to local camera: {camera_index}")

        if not stream.connect():
            logger.error("Failed to connect to video stream")
            return False

        logger.info("✓ Connected to video stream")
        logger.info("Press 'q' to quit, 's' to save a frame")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                success, frame = stream.read_frame()
                if success and frame is not None:
                    frame_count += 1

                    # Display frame
                    cv2.imshow("Video Stream Test", frame)

                    # Calculate and display FPS
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        fps = frame_count / elapsed
                        cv2.putText(
                            frame,
                            f"FPS: {fps:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow("Video Stream Test", frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        filename = f"captured_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        logger.info(f"Saved frame to: {filename}")

                else:
                    logger.warning("Failed to read frame")

        except KeyboardInterrupt:
            logger.info("\nStopped by user")

        finally:
            cv2.destroyAllWindows()
            stream.disconnect()

        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        logger.info(f"\nTotal frames: {frame_count}")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info("✓ Display test completed")
        return True

    except Exception as e:
        logger.error(f"Error in display test: {e}")
        return False


def main():
    """Run video stream tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test video streaming from phone to PC"
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default=None,
        help="IP Webcam URL (default: from config file). "
        "Note: Some IP Webcam apps may require /video suffix.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Local camera index (default: 0)",
    )
    parser.add_argument(
        "--test-local",
        action="store_true",
        help="Test local camera",
    )
    parser.add_argument(
        "--test-display",
        action="store_true",
        help="Test with OpenCV display window",
    )

    args = parser.parse_args()

    # Get default URL from config if not provided and IP Webcam test is requested
    if args.stream_url is None and (args.test_display or not args.test_local):
        try:
            # Load config inline (no helper function)
            import os
            from pathlib import Path

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
            logger.warning(f"Failed to load config: {e}")
            logger.info("Please provide --stream-url explicitly for IP Webcam tests")

    results = []

    # Test local camera
    if args.test_local:
        results.append(("Local Camera", test_local_camera()))

    # Test IP Webcam
    if args.stream_url:
        if args.test_display:
            results.append(
                ("IP Webcam (Display)", test_video_stream_with_display(args.stream_url))
            )
        else:
            results.append(("IP Webcam", test_ip_webcam(args.stream_url)))
    elif args.test_display:
        results.append(
            (
                "Local Camera (Display)",
                test_video_stream_with_display(None, args.camera_index),
            )
        )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    return 0 if all(r for _, r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
