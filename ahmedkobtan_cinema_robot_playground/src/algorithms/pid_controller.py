"""PID controller for visual servoing."""

from typing import Tuple


class PIDController:
    """Proportional-Integral-Derivative controller."""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_min: float = -255.0,
        output_max: float = 255.0,
        integral_limit: float = 100.0,
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output value
            output_max: Maximum output value
            integral_limit: Maximum integral accumulation (anti-windup)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit

        # State
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def compute(self, error: float, dt: float = 0.033) -> float:
        """
        Compute PID output.

        Args:
            error: Current error value
            dt: Time step (default 0.033s for 30 FPS)

        Returns:
            PID output value
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup)
        self.integral += error * dt
        self.integral = max(
            -self.integral_limit, min(self.integral_limit, self.integral)
        )
        i_term = self.ki * self.integral

        # Derivative term
        if self.last_time is not None:
            d_error = (error - self.last_error) / dt
        else:
            d_error = 0.0
        d_term = self.kd * d_error

        # Compute output
        output = p_term + i_term + d_term

        # Clamp output
        output = max(self.output_min, min(self.output_max, output))

        # Update state
        self.last_error = error
        self.last_time = dt

        return output

    def reset(self) -> None:
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """
        Update PID gains.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd


class DualAxisPIDController:
    """Dual-axis PID controller for X and Z axes."""

    def __init__(
        self,
        kp_x: float = 1.0,
        ki_x: float = 0.0,
        kd_x: float = 0.0,
        kp_z: float = 1.0,
        ki_z: float = 0.0,
        kd_z: float = 0.0,
    ):
        """
        Initialize dual-axis PID controller.

        Args:
            kp_x: Proportional gain for X-axis (servo)
            ki_x: Integral gain for X-axis
            kd_x: Derivative gain for X-axis
            kp_z: Proportional gain for Z-axis (wheels)
            ki_z: Integral gain for Z-axis
            kd_z: Derivative gain for Z-axis
        """
        self.pid_x = PIDController(
            kp=kp_x, ki=ki_x, kd=kd_x, output_min=-600, output_max=600
        )  # Servo range: ±600μs from center
        self.pid_z = PIDController(
            kp=kp_z, ki=ki_z, kd=kd_z, output_min=-255, output_max=255
        )  # Motor range: 0-255

    def compute(
        self, error_x: float, error_z: float, dt: float = 0.033
    ) -> Tuple[float, float]:
        """
        Compute PID outputs for both axes.

        Args:
            error_x: X-axis error (horizontal position)
            error_z: Z-axis error (depth/area)
            dt: Time step

        Returns:
            Tuple of (servo_output, wheel_output)
        """
        servo_output = self.pid_x.compute(error_x, dt)
        wheel_output = self.pid_z.compute(error_z, dt)

        return servo_output, wheel_output

    def reset(self) -> None:
        """Reset both controllers."""
        self.pid_x.reset()
        self.pid_z.reset()
