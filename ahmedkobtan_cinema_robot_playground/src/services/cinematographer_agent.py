"""Cinematographer Agent (Layer 2) - Trajectory planning."""

from typing import Dict, Optional

from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.algorithms.shot_planner import (
    ShotPlanner,
    ShotType,
)
from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
    MockTracker,
    SimpleTracker,
    Tracker,
    TrackingState,
)


class CinematographerAgent:
    """Cinematographer Agent for tracking and trajectory planning."""

    def __init__(
        self,
        tracker: Optional[Tracker] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        device: Optional[str] = None,
    ):
        """
        Initialize Cinematographer Agent.

        Args:
            tracker: Tracker to use (None for auto-select)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device
        self.tracker = tracker or self._create_tracker()
        self.shot_planner = ShotPlanner(frame_width, frame_height)
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Current state
        self.current_shot_type: Optional[ShotType] = None
        self.tracking_state: Optional[TrackingState] = None

    def _create_tracker(self) -> Tracker:
        """Create tracker (try real tracker, fallback to simple/mock)."""
        try:
            # Try Bot-SORT first
            from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
                BotSORTTracker,
            )

            tracker = BotSORTTracker(device=self.device)
            if tracker.is_available():
                logger.info("Using Bot-SORT tracker")
                return tracker
        except Exception:
            pass

        try:
            # Fallback to simple tracker
            logger.info("Using simple tracker")
            return SimpleTracker()
        except Exception:
            logger.warning("Using mock tracker")
            return MockTracker()

    def initialize_tracking(self, frame, initial_bbox: tuple) -> bool:
        """
        Initialize tracking with initial bounding box.

        Args:
            frame: Current video frame
            initial_bbox: Initial bounding box (x, y, width, height)

        Returns:
            True if initialization successful
        """
        try:
            self.tracking_state = self.tracker.update(frame, initial_bbox)
            return self.tracking_state is not None
        except Exception as e:
            logger.error(f"Error initializing tracking: {e}")
            return False

    def update_tracking(self, frame) -> Optional[TrackingState]:
        """
        Update tracking with new frame.

        Args:
            frame: Current video frame

        Returns:
            Tracking state or None if tracking lost
        """
        try:
            self.tracking_state = self.tracker.update(frame)
            return self.tracking_state
        except Exception as e:
            logger.error(f"Error updating tracking: {e}")
            return None

    def plan_trajectory(
        self,
        shot_type: str,
        tracking_state: Optional[TrackingState] = None,
    ) -> Dict:
        """
        Plan trajectory for shot type.

        Args:
            shot_type: Type of shot (follow, dolly_in, dolly_out, orbit, pan)
            tracking_state: Current tracking state (uses self.tracking_state if None)

        Returns:
            Dictionary with trajectory parameters
        """
        if tracking_state is None:
            tracking_state = self.tracking_state

        if tracking_state is None:
            logger.warning("No tracking state available for trajectory planning")
            return {}

        # Convert shot type string to enum
        try:
            shot_type_enum = ShotType(shot_type.lower())
        except ValueError:
            logger.warning(f"Unknown shot type: {shot_type}, using follow")
            shot_type_enum = ShotType.FOLLOW

        self.current_shot_type = shot_type_enum

        # Extract object info from tracking state
        x, y, w, h = tracking_state.bbox
        object_center_x = x + w / 2
        object_center_y = y + h / 2

        # Plan trajectory
        trajectory = self.shot_planner.plan_shot(
            shot_type_enum,
            object_center_x,
            object_center_y,
            w,
            h,
        )

        return trajectory

    def reset(self) -> None:
        """Reset agent state."""
        if self.tracker is not None:
            self.tracker.reset()
        self.shot_planner.reset()
        self.current_shot_type = None
        self.tracking_state = None
