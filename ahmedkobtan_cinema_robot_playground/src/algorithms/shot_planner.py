"""Shot planner for cinematic shot types."""

import math
from enum import Enum
from typing import Dict

from loguru import logger


class ShotType(Enum):
    """Cinematic shot types."""

    FOLLOW = "follow"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    ORBIT = "orbit"
    PAN = "pan"


class ShotPlanner:
    """Plans trajectories for cinematic shot types."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        """
        Initialize shot planner.

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center_x = frame_width / 2
        self.frame_center_y = frame_height / 2

        # Orbit parameters
        self.orbit_radius = 1.0  # meters (will be adjusted based on object distance)
        self.orbit_angle = 0.0  # Current angle in radians
        self.orbit_speed = 0.1  # radians per frame

        # Dolly parameters
        self.dolly_speed = 0.05  # meters per frame

    def plan_follow(
        self,
        object_center_x: float,
        object_center_y: float,
        object_width: float,
        object_height: float,
    ) -> Dict[str, float]:
        """
        Plan Follow shot - keep object centered.

        Args:
            object_center_x: Object center X coordinate
            object_center_y: Object center Y coordinate
            object_width: Object bounding box width
            object_height: Object bounding box height

        Returns:
            Dictionary with trajectory parameters
        """
        return {
            "shot_type": "follow",
            "target_x": object_center_x,
            "target_y": object_center_y,
            "target_area": object_width * object_height,
        }

    def plan_dolly_in(
        self,
        object_center_x: float,
        object_center_y: float,
        object_width: float,
        object_height: float,
    ) -> Dict[str, float]:
        """
        Plan Dolly In shot - move forward toward object.

        Args:
            object_center_x: Object center X coordinate
            object_center_y: Object center Y coordinate
            object_width: Object bounding box width
            object_height: Object bounding box height

        Returns:
            Dictionary with trajectory parameters
        """
        current_area = object_width * object_height
        target_area = current_area * 1.5  # Increase area by 50%

        return {
            "shot_type": "dolly_in",
            "target_x": object_center_x,
            "target_y": object_center_y,
            "target_area": target_area,
            "forward_speed": self.dolly_speed,
        }

    def plan_dolly_out(
        self,
        object_center_x: float,
        object_center_y: float,
        object_width: float,
        object_height: float,
    ) -> Dict[str, float]:
        """
        Plan Dolly Out shot - move backward away from object.

        Args:
            object_center_x: Object center X coordinate
            object_center_y: Object center Y coordinate
            object_width: Object bounding box width
            object_height: Object bounding box height

        Returns:
            Dictionary with trajectory parameters
        """
        current_area = object_width * object_height
        target_area = current_area * 0.7  # Decrease area by 30%

        return {
            "shot_type": "dolly_out",
            "target_x": object_center_x,
            "target_y": object_center_y,
            "target_area": target_area,
            "backward_speed": self.dolly_speed,
        }

    def plan_orbit(
        self,
        object_center_x: float,
        object_center_y: float,
        object_width: float,
        object_height: float,
    ) -> Dict[str, float]:
        """
        Plan Orbit shot - circle around object.

        Args:
            object_center_x: Object center X coordinate
            object_center_y: Object center Y coordinate
            object_width: Object bounding box width
            object_height: Object bounding box height

        Returns:
            Dictionary with trajectory parameters
        """
        # Update orbit angle
        self.orbit_angle += self.orbit_speed
        if self.orbit_angle >= 2 * math.pi:
            self.orbit_angle -= 2 * math.pi

        # Calculate circular path
        # For now, we'll simulate this by adjusting the target position
        # In a real implementation, this would coordinate wheels + servo

        return {
            "shot_type": "orbit",
            "target_x": object_center_x,  # Keep object centered
            "target_y": object_center_y,
            "target_area": object_width * object_height,  # Maintain distance
            "orbit_angle": self.orbit_angle,
            "orbit_radius": self.orbit_radius,
        }

    def plan_shot(
        self,
        shot_type: ShotType,
        object_center_x: float,
        object_center_y: float,
        object_width: float,
        object_height: float,
    ) -> Dict[str, float]:
        """
        Plan shot based on shot type.

        Args:
            shot_type: Type of shot to plan
            object_center_x: Object center X coordinate
            object_center_y: Object center Y coordinate
            object_width: Object bounding box width
            object_height: Object bounding box height

        Returns:
            Dictionary with trajectory parameters
        """
        if shot_type == ShotType.FOLLOW:
            return self.plan_follow(
                object_center_x, object_center_y, object_width, object_height
            )
        elif shot_type == ShotType.DOLLY_IN:
            return self.plan_dolly_in(
                object_center_x, object_center_y, object_width, object_height
            )
        elif shot_type == ShotType.DOLLY_OUT:
            return self.plan_dolly_out(
                object_center_x, object_center_y, object_width, object_height
            )
        elif shot_type == ShotType.ORBIT:
            return self.plan_orbit(
                object_center_x, object_center_y, object_width, object_height
            )
        elif shot_type == ShotType.PAN:
            return self.plan_follow(
                object_center_x, object_center_y, object_width, object_height
            )  # Pan is similar to follow
        else:
            logger.warning(f"Unknown shot type: {shot_type}")
            return self.plan_follow(
                object_center_x, object_center_y, object_width, object_height
            )

    def reset(self) -> None:
        """Reset planner state."""
        self.orbit_angle = 0.0
