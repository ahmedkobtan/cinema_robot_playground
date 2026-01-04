"""Test video stream connection logic."""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ahmedkobtan_cinema_robot_playground.src.services.video_stream import VideoStream


class TestVideoStreamConnection(unittest.TestCase):
    """Test VideoStream connection handling."""

    @patch("cv2.VideoCapture")
    def test_connect_with_url(self, mock_video_capture):
        """Test connection with URL."""
        # Mock successful connection on first attempt
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        stream = VideoStream(stream_url="http://192.168.0.220:8080")
        result = stream.connect()

        self.assertTrue(result)
        self.assertTrue(stream.is_streaming)
        # Should be called with URL and FFMPEG backend
        mock_video_capture.assert_called_with(
            "http://192.168.0.220:8080", cv2.CAP_FFMPEG
        )
        mock_cap.isOpened.assert_called()
        mock_cap.read.assert_called()
        # Cleanup
        stream.disconnect()

    @patch("cv2.VideoCapture")
    def test_connect_with_url_video_suffix(self, mock_video_capture):
        """Test connection with URL that includes /video suffix."""
        # Mock successful connection - tries base URL first, then /video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        stream = VideoStream(stream_url="http://192.168.0.220:8080/video")
        result = stream.connect()

        self.assertTrue(result)
        self.assertTrue(stream.is_streaming)
        # Should try base URL first (without /video), then /video URL
        # Check that it was called with FFMPEG backend
        calls = mock_video_capture.call_args_list
        self.assertGreater(len(calls), 0)
        # First call should be base URL (without /video) with FFMPEG backend
        self.assertEqual(calls[0][0][0], "http://192.168.0.220:8080")
        self.assertEqual(calls[0][0][1], cv2.CAP_FFMPEG)
        # Cleanup
        stream.disconnect()

    @patch("cv2.VideoCapture")
    def test_connect_fails_when_capture_not_opened(self, mock_video_capture):
        """Test connection fails when VideoCapture is not opened."""
        # Mock failed connection
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        stream = VideoStream(stream_url="http://192.168.0.220:8080")
        result = stream.connect()

        self.assertFalse(result)
        self.assertFalse(stream.is_streaming)
        mock_cap.read.assert_not_called()
        # Cleanup
        stream.disconnect()

    @patch("cv2.VideoCapture")
    def test_connect_fails_when_read_fails(self, mock_video_capture):
        """Test connection fails when initial read fails."""
        # Mock connection opened but read fails for all URL attempts
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        stream = VideoStream(stream_url="http://192.168.0.220:8080")
        result = stream.connect()

        self.assertFalse(result)
        self.assertFalse(stream.is_streaming)
        # read() is called for each URL attempt (original URL, then /video, then default backend)
        # Should be called at least once, possibly multiple times
        self.assertGreaterEqual(mock_cap.read.call_count, 1)
        # Cleanup
        stream.disconnect()

    @patch("cv2.VideoCapture")
    def test_connect_handles_exception(self, mock_video_capture):
        """Test connection handles exceptions gracefully."""
        # Mock exception during connection
        mock_video_capture.side_effect = Exception("Connection error")

        stream = VideoStream(stream_url="http://192.168.0.220:8080")
        result = stream.connect()

        self.assertFalse(result)
        self.assertFalse(stream.is_streaming)
        # Cleanup
        stream.disconnect()

    def test_url_preserved_exactly_as_provided(self):
        """Test that URL is preserved exactly as provided (no automatic /video append)."""
        test_urls = [
            "http://192.168.0.220:8080",
            "http://192.168.0.220:8080/video",
            "http://192.168.0.220:8080/stream",
            "https://192.168.0.220:8080",
        ]

        for url in test_urls:
            with self.subTest(url=url):
                stream = VideoStream(stream_url=url)
                self.assertEqual(stream.stream_url, url)
                # Cleanup
                stream.disconnect()

    @patch("cv2.VideoCapture")
    def test_connect_uses_exact_url_provided(self, mock_video_capture):
        """Test that connect() tries the exact URL provided first, then alternatives."""
        # Mock successful connection on first attempt
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        # Test with URL without /video
        test_url = "http://192.168.0.220:8080"
        stream = VideoStream(stream_url=test_url)
        result = stream.connect()

        self.assertTrue(result)
        # Verify VideoCapture was called with URL and FFMPEG backend
        # First call should be the original URL
        calls = mock_video_capture.call_args_list
        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[0][0][0], test_url)
        self.assertEqual(calls[0][0][1], cv2.CAP_FFMPEG)
        stream.disconnect()

        # Test with URL with /video
        test_url_with_video = "http://192.168.0.220:8080/video"
        stream2 = VideoStream(stream_url=test_url_with_video)
        mock_video_capture.reset_mock()
        mock_video_capture.return_value = mock_cap
        result2 = stream2.connect()

        self.assertTrue(result2)
        # First call should try base URL (without /video), then /video URL
        calls = mock_video_capture.call_args_list
        self.assertGreater(len(calls), 0)
        # First attempt is base URL (without /video)
        self.assertEqual(calls[0][0][0], "http://192.168.0.220:8080")
        self.assertEqual(calls[0][0][1], cv2.CAP_FFMPEG)
        stream2.disconnect()


if __name__ == "__main__":
    unittest.main()
