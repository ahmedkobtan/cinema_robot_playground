"""Models package for cinema robot."""

from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    BoundingBox,
    DetectionModel,
    GroundingDINOModel,
    MockDetectionModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import (
    FollowAnythingFallback,
    FollowAnythingModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
    BotSORTTracker,
    MockTracker,
    SimpleTracker,
    Tracker,
    TrackingState,
)

__all__ = [
    "BoundingBox",
    "DetectionModel",
    "GroundingDINOModel",
    "MockDetectionModel",
    "FollowAnythingModel",
    "FollowAnythingFallback",
    "Tracker",
    "TrackingState",
    "BotSORTTracker",
    "SimpleTracker",
    "MockTracker",
]
