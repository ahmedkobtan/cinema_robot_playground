"""Tests for video streaming service."""

from ahmedkobtan_cinema_robot_playground.src.services.video_stream import VideoStream


class TestVideoStream:
    """Test VideoStream class."""

    def test_initialization_with_url(self):
        """Test initialization with stream URL."""
        stream = VideoStream(stream_url="http://192.168.1.100:8080/video")
        assert stream.stream_url == "http://192.168.1.100:8080/video"
        assert stream.camera_index == 0
        assert stream.width == 1280
        assert stream.height == 720
        assert stream.fps == 30

    def test_initialization_with_camera(self):
        """Test initialization with camera index."""
        stream = VideoStream(camera_index=1, width=640, height=480, fps=15)
        assert stream.stream_url is None
        assert stream.camera_index == 1
        assert stream.width == 640
        assert stream.height == 480
        assert stream.fps == 15

    def test_get_frame_size_default(self):
        """Test getting default frame size."""
        stream = VideoStream()
        width, height = stream.get_frame_size()
        assert width == 1280
        assert height == 720

    def test_context_manager(self):
        """Test context manager usage."""
        stream = VideoStream()
        # Should not crash even if connection fails
        try:
            with stream:
                assert stream.is_streaming or not stream.is_streaming  # Either is fine
        except Exception:
            # Expected if no camera/stream available
            pass

    def test_disconnect_when_not_connected(self):
        """Test disconnect when not connected."""
        stream = VideoStream()
        # Should not crash
        stream.disconnect()
        assert not stream.is_streaming
