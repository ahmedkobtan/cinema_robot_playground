"""Tests for visual servoing."""

from ahmedkobtan_cinema_robot_playground.src.algorithms.visual_servoing import (
    VisualServoingController,
)


class TestVisualServoingController:
    """Test visual servoing controller."""

    def test_initialization(self):
        """Test controller initialization."""
        controller = VisualServoingController(frame_width=1280, frame_height=720)
        assert controller.frame_width == 1280
        assert controller.frame_height == 720
        assert controller.frame_center_x == 640
        assert controller.frame_center_y == 360

    def test_compute_errors(self):
        """Test error computation."""
        controller = VisualServoingController(frame_width=1280, frame_height=720)
        error_x, error_z = controller.compute_errors(
            object_center_x=700.0,  # Right of center
            object_center_y=360.0,
            object_width=100.0,
            object_height=100.0,
        )
        assert error_x > 0  # Positive error (object to the right)
        assert error_z != 0  # Area-based error

    def test_compute_commands(self):
        """Test command computation."""
        controller = VisualServoingController(frame_width=1280, frame_height=720)
        servo_pulse, wheel_left, wheel_right, wheel_forward = (
            controller.compute_commands(
                object_center_x=700.0,
                object_center_y=360.0,
                object_width=100.0,
                object_height=100.0,
            )
        )
        # Servo pulse should be in valid range
        assert 900 <= servo_pulse <= 2100
        # Wheel speeds should be in valid range
        assert 0 <= wheel_left <= 255
        assert 0 <= wheel_right <= 255
        assert -255 <= wheel_forward <= 255

    def test_movement_hierarchy(self):
        """Test movement hierarchy (servo first)."""
        controller = VisualServoingController(
            frame_width=1280,
            frame_height=720,
            servo_threshold=0.1,  # 10% threshold
        )
        # Small error should use servo only
        servo_pulse, wheel_left, wheel_right, wheel_forward = (
            controller.compute_commands(
                object_center_x=650.0,  # Small offset
                object_center_y=360.0,
                object_width=100.0,
                object_height=100.0,
            )
        )
        # Should primarily use servo
        assert servo_pulse != 1500  # Servo moved
        # Wheels might be used if needed, but servo is primary

    def test_reset(self):
        """Test controller reset."""
        controller = VisualServoingController()
        controller.compute_commands(700.0, 360.0, 100.0, 100.0)
        controller.reset()
        assert controller.current_servo_pulse == 1500  # Back to center
