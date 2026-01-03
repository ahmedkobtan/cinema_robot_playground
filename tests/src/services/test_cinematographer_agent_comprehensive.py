"""Comprehensive tests for Cinematographer Agent - matches script-level testing."""

import numpy as np

from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import MockTracker
from ahmedkobtan_cinema_robot_playground.src.services.cinematographer_agent import (
    CinematographerAgent,
)


class TestCinematographerAgentComprehensive:
    """Comprehensive tests matching scripts/test_setup.py."""

    def test_initialization_with_device(self):
        """Test agent initialization with device parameter."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        assert agent.device == "cpu"
        assert agent.tracker is not None
        assert agent.shot_planner is not None

    def test_initialization_auto_tracker(self):
        """Test agent initialization with auto-tracker selection."""
        agent = CinematographerAgent(device="cpu")
        # Should create a tracker (may be Bot-SORT, Simple, or Mock)
        assert agent.tracker is not None

    def test_initialize_tracking(self):
        """Test tracking initialization."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        success = agent.initialize_tracking(frame, initial_bbox)
        assert success is True

    def test_update_tracking(self):
        """Test tracking update."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        tracking_state = agent.update_tracking(frame)
        assert tracking_state is not None
        assert hasattr(tracking_state, "bbox")
        assert hasattr(tracking_state, "confidence")

    def test_plan_trajectory_follow(self):
        """Test trajectory planning for follow shot."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        trajectory = agent.plan_trajectory("follow")
        assert trajectory is not None
        assert "shot_type" in trajectory
        assert trajectory["shot_type"] == "follow"

    def test_plan_trajectory_orbit(self):
        """Test trajectory planning for orbit shot."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        trajectory = agent.plan_trajectory("orbit")
        assert trajectory is not None
        assert trajectory["shot_type"] == "orbit"

    def test_plan_trajectory_dolly_in(self):
        """Test trajectory planning for dolly in shot."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        trajectory = agent.plan_trajectory("dolly_in")
        assert trajectory is not None
        assert trajectory["shot_type"] == "dolly_in"

    def test_plan_trajectory_dolly_out(self):
        """Test trajectory planning for dolly out shot."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        trajectory = agent.plan_trajectory("dolly_out")
        assert trajectory is not None
        assert trajectory["shot_type"] == "dolly_out"

    def test_plan_trajectory_pan(self):
        """Test trajectory planning for pan shot."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        trajectory = agent.plan_trajectory("pan")
        assert trajectory is not None
        # Pan may map to follow if not explicitly implemented
        assert trajectory["shot_type"] in ["pan", "follow"]

    def test_plan_trajectory_without_tracking(self):
        """Test trajectory planning without tracking state."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        # Don't initialize tracking
        trajectory = agent.plan_trajectory("follow")
        # Should return empty dict or handle gracefully
        assert isinstance(trajectory, dict)

    def test_reset(self):
        """Test agent reset."""
        agent = CinematographerAgent(tracker=MockTracker(), device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        agent.reset()
        assert agent.tracking_state is None
        assert agent.current_shot_type is None

    def test_device_parameter_consistency(self):
        """Test that device parameter is consistently used."""
        for device in ["cpu", "cuda"]:
            agent = CinematographerAgent(tracker=MockTracker(), device=device)
            assert agent.device == device
            # Tracker should also use device if it supports it
            if hasattr(agent.tracker, "device"):
                assert agent.tracker.device == device
