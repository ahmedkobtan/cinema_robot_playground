"""Visual servoing controller with movement hierarchy."""

from typing import Tuple

from ahmedkobtan_cinema_robot_playground.src.algorithms.pid_controller import (
    DualAxisPIDController,
)


class VisualServoingController:
    """Visual servoing controller with servo-first movement hierarchy."""

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        servo_center_pulse: int = 1500,
        servo_min_pulse: int = 900,
        servo_max_pulse: int = 2100,
        servo_threshold: float = 0.2,  # 20% of frame width
        kp_x: float = 2.0,
        ki_x: float = 0.0,
        kd_x: float = 0.5,
        kp_z: float = 0.5,
        ki_z: float = 0.0,
        kd_z: float = 0.1,
    ):
        """
        Initialize visual servoing controller.

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            servo_center_pulse: Center servo pulse width (microseconds)
            servo_min_pulse: Minimum servo pulse width
            servo_max_pulse: Maximum servo pulse width
            servo_threshold: Threshold for using wheels (fraction of frame width)
            kp_x: Proportional gain for X-axis (servo)
            ki_x: Integral gain for X-axis
            kd_x: Derivative gain for X-axis
            kp_z: Proportional gain for Z-axis (wheels)
            ki_z: Integral gain for Z-axis
            kd_z: Derivative gain for Z-axis
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center_x = frame_width / 2
        self.frame_center_y = frame_height / 2

        self.servo_center_pulse = servo_center_pulse
        self.servo_min_pulse = servo_min_pulse
        self.servo_max_pulse = servo_max_pulse
        self.servo_range = servo_max_pulse - servo_min_pulse
        self.servo_threshold = servo_threshold * frame_width  # Convert to pixels

        # Desired bounding box area (10% of frame)
        self.desired_area = frame_width * frame_height * 0.1

        # PID controllers
        self.pid_controller = DualAxisPIDController(
            kp_x=kp_x,
            ki_x=ki_x,
            kd_x=kd_x,
            kp_z=kp_z,
            ki_z=ki_z,
            kd_z=kd_z,
        )

        # Current servo position
        self.current_servo_pulse = servo_center_pulse

    def compute_errors(
        self,
        object_center_x: float,
        object_center_y: float,
        object_width: float,
        object_height: float,
    ) -> Tuple[float, float]:
        """
        Compute X and Z axis errors.

        Args:
            object_center_x: Object center X coordinate
            object_center_y: Object center Y coordinate
            object_width: Object bounding box width
            object_height: Object bounding box height

        Returns:
            Tuple of (error_x, error_z)
        """
        # X-axis error (horizontal)
        error_x = object_center_x - self.frame_center_x

        # Z-axis error (depth based on area)
        current_area = object_width * object_height
        error_z = current_area - self.desired_area

        return error_x, error_z

    def compute_commands(
        self,
        object_center_x: float,
        object_center_y: float,
        object_width: float,
        object_height: float,
        dt: float = 0.033,
    ) -> Tuple[int, int, int, int]:
        """
        Compute motor commands using movement hierarchy.

        Args:
            object_center_x: Object center X coordinate
            object_center_y: Object center Y coordinate
            object_width: Object bounding box width
            object_height: Object bounding box height
            dt: Time step (default 0.033s for 30 FPS)

        Returns:
            Tuple of (servo_pulse, wheel_left_speed, wheel_right_speed, wheel_forward_speed)
            All speeds are 0-255, servo_pulse is 900-2100 microseconds
        """
        # Compute errors
        error_x, error_z = self.compute_errors(
            object_center_x, object_center_y, object_width, object_height
        )

        # Compute PID outputs
        servo_output, wheel_output = self.pid_controller.compute(error_x, error_z, dt)

        # Movement hierarchy: Servo first (Tier 1)
        servo_pulse = int(self.servo_center_pulse + servo_output)
        servo_pulse = max(self.servo_min_pulse, min(self.servo_max_pulse, servo_pulse))
        self.current_servo_pulse = servo_pulse

        # Check if servo is at limit or error is too large
        servo_at_limit = (
            servo_pulse <= self.servo_min_pulse + 50
            or servo_pulse >= self.servo_max_pulse - 50
        )
        error_too_large = abs(error_x) > self.servo_threshold

        # Tier 2: Use wheels if needed
        if servo_at_limit or error_too_large:
            # Use wheels for coarse control
            if error_x > 0:
                # Object to the right, turn right
                wheel_left_speed = int(abs(wheel_output))
                wheel_right_speed = 0
            else:
                # Object to the left, turn left
                wheel_left_speed = 0
                wheel_right_speed = int(abs(wheel_output))

            # Z-axis control (forward/backward)
            if error_z > 0:
                # Object too close, move backward
                wheel_forward_speed = -int(abs(wheel_output))
            else:
                # Object too far, move forward
                wheel_forward_speed = int(abs(wheel_output))
        else:
            # Servo only, no wheel movement
            wheel_left_speed = 0
            wheel_right_speed = 0
            wheel_forward_speed = 0

        # Clamp speeds
        wheel_left_speed = max(0, min(255, abs(wheel_left_speed)))
        wheel_right_speed = max(0, min(255, abs(wheel_right_speed)))
        wheel_forward_speed = max(-255, min(255, wheel_forward_speed))

        return servo_pulse, wheel_left_speed, wheel_right_speed, wheel_forward_speed

    def reset(self) -> None:
        """Reset controller state."""
        self.pid_controller.reset()
        self.current_servo_pulse = self.servo_center_pulse

    def set_desired_area(self, area: float) -> None:
        """
        Set desired bounding box area.

        Args:
            area: Desired area in pixels^2
        """
        self.desired_area = area
