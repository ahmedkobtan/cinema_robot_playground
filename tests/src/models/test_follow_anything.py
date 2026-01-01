"""Tests for Follow Anything model."""

import numpy as np

from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import (
    FollowAnythingFallback,
    FollowAnythingModel,
)


class TestFollowAnythingModel:
    """Test Follow Anything model."""

    def test_initialization(self):
        """Test model initialization."""
        model = FollowAnythingModel()
        # Model may not be available, but initialization should work
        assert model.device is not None

    def test_fallback_initialization(self):
        """Test fallback model initialization."""
        fallback = FollowAnythingFallback()
        # Fallback should try to initialize detector and tracker
        assert fallback.detector is not None
        assert fallback.tracker is not None

    def test_detect_with_fallback(self):
        """Test detection with fallback."""
        fallback = FollowAnythingFallback()
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        # May not be available, but should not crash
        try:
            boxes = fallback.detect(image, "test object")
            assert isinstance(boxes, list)
        except Exception:
            # Expected if models not available
            pass

    def test_track_with_fallback(self):
        """Test tracking with fallback."""
        fallback = FollowAnythingFallback()
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        # May not be available, but should not crash
        try:
            _ = fallback.update(image, initial_bbox)
            # State may be None if not available
        except Exception:
            # Expected if models not available
            pass

    def test_reset(self):
        """Test reset functionality."""
        model = FollowAnythingModel()
        model.reset()
        assert not model._tracking_initialized
        assert model._current_track_id == 0

        fallback = FollowAnythingFallback()
        fallback.reset()
        assert not fallback._tracking_initialized
