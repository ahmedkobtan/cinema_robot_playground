"""Video streaming service for camera input."""

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class VideoStream:
    """Handles video stream from smartphone camera via Wi-Fi."""

    def __init__(
        self,
        stream_url: Optional[str] = None,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        Initialize video stream.

        Args:
            stream_url: URL for IP Webcam/DroidCam stream (e.g., "http://192.168.1.100:8080/video")
            camera_index: Local camera index if using USB camera
            width: Frame width (default 1280 for 720p)
            height: Frame height (default 720 for 720p)
            fps: Target frames per second
        """
        self.stream_url = stream_url
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_streaming = False

    def connect(self) -> bool:
        """
        Connect to video stream.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.stream_url:
                logger.info(f"Connecting to stream: {self.stream_url}")

                # Try different URL formats for IP Webcam
                urls_to_try = [self.stream_url]

                # If URL doesn't end with /video, try adding it
                if not self.stream_url.endswith(
                    "/video"
                ) and not self.stream_url.endswith("/video/"):
                    urls_to_try.append(f"{self.stream_url.rstrip('/')}/video")

                # Also try without /video if original had it
                if self.stream_url.endswith("/video"):
                    base_url = self.stream_url.rsplit("/video", 1)[0]
                    urls_to_try.insert(0, base_url)

                # Remove duplicates while preserving order
                seen = set()
                urls_to_try = [
                    url for url in urls_to_try if url not in seen and not seen.add(url)
                ]

                last_error = None
                for url in urls_to_try:
                    try:
                        logger.debug(f"Trying URL: {url}")
                        # Use FFMPEG backend for HTTP streams (more reliable)
                        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                        # Set buffer size to reduce premature stream ending
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        if self.cap.isOpened():
                            # Test read with timeout
                            ret, frame = self.cap.read()
                            if ret and frame is not None:
                                logger.info(f"✓ Connected successfully using: {url}")
                                if url != self.stream_url:
                                    logger.info(f"  (Original URL: {self.stream_url})")
                                self.is_streaming = True
                                return True
                            else:
                                self.cap.release()
                                self.cap = None
                                last_error = f"Failed to read frame from {url}"
                        else:
                            last_error = f"Failed to open stream: {url}"
                    except Exception as e:
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None
                        last_error = f"Error with {url}: {e}"
                        continue

                # If all URLs failed, try with default backend
                logger.warning("FFMPEG backend failed, trying default backend")
                try:
                    self.cap = cv2.VideoCapture(self.stream_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            logger.info("✓ Connected using default backend")
                            self.is_streaming = True
                            return True
                except Exception as e:
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    last_error = f"Default backend also failed: {e}"

                logger.error(
                    f"Failed to connect to video stream. Last error: {last_error}"
                )
                logger.info("\nTroubleshooting:")
                logger.info("1. Verify IP Webcam app is running on your phone")
                logger.info("2. Check that phone and PC are on the same Wi-Fi network")
                logger.info("3. Try opening the URL in a web browser:")
                logger.info(f"   - {self.stream_url}")
                if not self.stream_url.endswith("/video"):
                    logger.info(f"   - {self.stream_url}/video")
                logger.info(
                    "4. Check IP Webcam app settings for the correct stream URL"
                )
                logger.info("5. The 'Stream ends prematurely' error often means:")
                logger.info("   - URL format is incorrect (try adding /video suffix)")
                logger.info("   - IP Webcam app needs to be restarted")
                logger.info("   - Network connection is unstable")
                return False
            else:
                logger.info(f"Connecting to local camera: {self.camera_index}")
                self.cap = cv2.VideoCapture(self.camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.cap.isOpened():
                logger.error("Failed to open video stream")
                return False

            # Test read
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read from video stream")
                return False

            self.is_streaming = True
            logger.info("Video stream connected successfully")
            return True

        except Exception as e:
            logger.error(f"Error connecting to video stream: {e}")
            if self.cap is not None:
                self.cap.release()
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from video stream.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if not self.is_streaming or self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get frame dimensions.

        Returns:
            Tuple of (width, height)
        """
        if self.cap is None:
            return self.width, self.height

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        except Exception:
            return self.width, self.height

    def disconnect(self) -> None:
        """Disconnect from video stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_streaming = False
        logger.info("Video stream disconnected")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
