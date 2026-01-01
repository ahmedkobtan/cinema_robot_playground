"""Tests for Cinematographer Agent."""

import numpy as np

from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import MockTracker
from ahmedkobtan_cinema_robot_playground.src.services.cinematographer_agent import (
    CinematographerAgent,
)


class TestCinematographerAgent:
    """Test Cinematographer Agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = CinematographerAgent(tracker=MockTracker())
        assert agent.tracker is not None
        assert agent.shot_planner is not None

    def test_initialize_tracking(self):
        """Test tracking initialization."""
        agent = CinematographerAgent(tracker=MockTracker())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        success = agent.initialize_tracking(frame, initial_bbox)
        assert success

    def test_update_tracking(self):
        """Test tracking update."""
        agent = CinematographerAgent(tracker=MockTracker())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        tracking_state = agent.update_tracking(frame)
        assert tracking_state is not None

    def test_plan_trajectory_follow(self):
        """Test trajectory planning for follow shot."""
        agent = CinematographerAgent(tracker=MockTracker())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        trajectory = agent.plan_trajectory("follow")
        assert trajectory is not None
        assert "shot_type" in trajectory

    def test_plan_trajectory_orbit(self):
        """Test trajectory planning for orbit shot."""
        agent = CinematographerAgent(tracker=MockTracker())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        trajectory = agent.plan_trajectory("orbit")
        assert trajectory is not None
        assert trajectory["shot_type"] == "orbit"

    def test_reset(self):
        """Test agent reset."""
        agent = CinematographerAgent(tracker=MockTracker())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        agent.initialize_tracking(frame, initial_bbox)
        agent.reset()
        assert agent.tracking_state is None
