"""Pilot Agent (Layer 3) - Visual servoing and motor control."""

from typing import Dict, Optional

from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.algorithms.visual_servoing import (
    VisualServoingController,
)
from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import TrackingState
from ahmedkobtan_cinema_robot_playground.src.utils.robot_commander import RobotCommander


class PilotAgent:
    """Pilot Agent for executing motor commands."""

    def __init__(
        self,
        robot_commander: RobotCommander,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):
        """
        Initialize Pilot Agent.

        Args:
            robot_commander: Robot commander for sending commands
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.robot_commander = robot_commander
        self.visual_servoing = VisualServoingController(frame_width, frame_height)

    def execute_trajectory(
        self,
        trajectory: Dict,
        tracking_state: Optional[TrackingState] = None,
    ) -> bool:
        """
        Execute trajectory by generating motor commands.

        Args:
            trajectory: Trajectory parameters from Cinematographer
            tracking_state: Current tracking state

        Returns:
            True if commands executed successfully
        """
        if tracking_state is None:
            logger.warning("No tracking state available")
            return False

        # Extract object info
        x, y, w, h = tracking_state.bbox
        object_center_x = x + w / 2
        object_center_y = y + h / 2

        # Get target from trajectory
        target_x = trajectory.get("target_x", object_center_x)
        target_y = trajectory.get("target_y", object_center_y)
        target_area = trajectory.get("target_area", w * h)

        # Update desired area if specified
        if "target_area" in trajectory:
            self.visual_servoing.set_desired_area(target_area)

        # Compute motor commands using visual servoing
        servo_pulse, wheel_left, wheel_right, wheel_forward = (
            self.visual_servoing.compute_commands(
                target_x,
                target_y,
                w,
                h,
            )
        )

        # Execute commands based on shot type
        shot_type = trajectory.get("shot_type", "follow")

        if shot_type == "dolly_in":
            # Forward movement
            if wheel_forward > 0:
                return self.robot_commander.move_forward(wheel_forward)
            else:
                return self.robot_commander.set_servo(servo_pulse)

        elif shot_type == "dolly_out":
            # Backward movement
            if wheel_forward < 0:
                return self.robot_commander.move_backward(abs(wheel_forward))
            else:
                return self.robot_commander.set_servo(servo_pulse)

        elif shot_type == "orbit":
            # Coordinated movement (simplified - can be enhanced)
            # For now, use servo to keep object centered while wheels move
            if wheel_left > 0 or wheel_right > 0:
                # Turn while maintaining servo
                if wheel_left > wheel_right:
                    self.robot_commander.turn_left(wheel_left)
                else:
                    self.robot_commander.turn_right(wheel_right)
            return self.robot_commander.set_servo(servo_pulse)

        else:
            # Follow or pan - use visual servoing output
            success = True

            # Set servo
            if not self.robot_commander.set_servo(servo_pulse):
                success = False

            # Use wheels if needed
            if wheel_left > 0:
                if not self.robot_commander.turn_left(wheel_left):
                    success = False
            elif wheel_right > 0:
                if not self.robot_commander.turn_right(wheel_right):
                    success = False

            # Forward/backward
            if wheel_forward > 0:
                if not self.robot_commander.move_forward(wheel_forward):
                    success = False
            elif wheel_forward < 0:
                if not self.robot_commander.move_backward(abs(wheel_forward)):
                    success = False

            return success

    def stop(self) -> bool:
        """Stop all robot movement."""
        return self.robot_commander.stop()

    def reset(self) -> None:
        """Reset agent state."""
        self.visual_servoing.reset()
        self.stop()
