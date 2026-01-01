"""Tests for PID controller."""

from ahmedkobtan_cinema_robot_playground.src.algorithms.pid_controller import (
    DualAxisPIDController,
    PIDController,
)


class TestPIDController:
    """Test PID controller."""

    def test_pid_initialization(self):
        """Test PID controller initialization."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.01)
        assert pid.kp == 1.0
        assert pid.ki == 0.1
        assert pid.kd == 0.01

    def test_pid_compute(self):
        """Test PID computation."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        output = pid.compute(10.0)
        assert output == 10.0  # P term only

    def test_pid_integral(self):
        """Test PID integral term."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.0)
        output1 = pid.compute(10.0, dt=0.1)
        output2 = pid.compute(10.0, dt=0.1)
        # Second output should be higher due to integral accumulation
        assert output2 > output1

    def test_pid_output_clamping(self):
        """Test PID output clamping."""
        pid = PIDController(kp=100.0, output_min=-10.0, output_max=10.0)
        output = pid.compute(100.0)
        assert -10.0 <= output <= 10.0

    def test_pid_reset(self):
        """Test PID reset."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.0)
        pid.compute(10.0, dt=0.1)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.last_error == 0.0


class TestDualAxisPIDController:
    """Test dual-axis PID controller."""

    def test_dual_axis_initialization(self):
        """Test dual-axis PID initialization."""
        controller = DualAxisPIDController(kp_x=1.0, kp_z=2.0)
        assert controller.pid_x.kp == 1.0
        assert controller.pid_z.kp == 2.0

    def test_dual_axis_compute(self):
        """Test dual-axis computation."""
        controller = DualAxisPIDController(kp_x=1.0, kp_z=2.0)
        servo_output, wheel_output = controller.compute(10.0, 5.0)
        assert servo_output == 10.0  # X-axis
        assert wheel_output == 10.0  # Z-axis (2.0 * 5.0)
