"""Test video stream connection logic."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ahmedkobtan_cinema_robot_playground.src.services.video_stream import VideoStream


class TestVideoStreamConnection(unittest.TestCase):
    """Test VideoStream connection handling."""

    @patch("cv2.VideoCapture")
    def test_connect_with_url(self, mock_video_capture):
        """Test connection with URL."""
        # Mock successful connection
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        stream = VideoStream(stream_url="http://192.168.0.220:8080")
        result = stream.connect()

        self.assertTrue(result)
        self.assertTrue(stream.is_streaming)
        mock_video_capture.assert_called_once_with("http://192.168.0.220:8080")
        mock_cap.isOpened.assert_called_once()
        mock_cap.read.assert_called_once()
        # Cleanup
        stream.disconnect()

    @patch("cv2.VideoCapture")
    def test_connect_with_url_video_suffix(self, mock_video_capture):
        """Test connection with URL that includes /video suffix."""
        # Mock successful connection
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        stream = VideoStream(stream_url="http://192.168.0.220:8080/video")
        result = stream.connect()

        self.assertTrue(result)
        self.assertTrue(stream.is_streaming)
        mock_video_capture.assert_called_once_with("http://192.168.0.220:8080/video")
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
        # Mock connection opened but read fails
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        stream = VideoStream(stream_url="http://192.168.0.220:8080")
        result = stream.connect()

        self.assertFalse(result)
        self.assertFalse(stream.is_streaming)
        mock_cap.read.assert_called_once()
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
        """Test that connect() uses the exact URL provided to VideoCapture."""
        # Mock successful connection
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        # Test with URL without /video
        test_url = "http://192.168.0.220:8080"
        stream = VideoStream(stream_url=test_url)
        result = stream.connect()

        self.assertTrue(result)
        # Verify VideoCapture was called with exact URL (no modifications)
        mock_video_capture.assert_called_once_with(test_url)
        stream.disconnect()

        # Test with URL with /video
        test_url_with_video = "http://192.168.0.220:8080/video"
        stream2 = VideoStream(stream_url=test_url_with_video)
        mock_video_capture.reset_mock()
        mock_video_capture.return_value = mock_cap
        result2 = stream2.connect()

        self.assertTrue(result2)
        # Verify VideoCapture was called with exact URL including /video
        mock_video_capture.assert_called_once_with(test_url_with_video)
        stream2.disconnect()


if __name__ == "__main__":
    unittest.main()
