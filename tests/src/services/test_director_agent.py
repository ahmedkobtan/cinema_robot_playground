"""Tests for Director Agent."""

import numpy as np

from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    MockDetectionModel,
)
from ahmedkobtan_cinema_robot_playground.src.services.director_agent import (
    DirectorAgent,
)


class TestDirectorAgent:
    """Test Director Agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = DirectorAgent(use_llm=False)
        assert agent.detection_model is not None

    def test_parse_command_simple(self):
        """Test simple command parsing."""
        agent = DirectorAgent(use_llm=False)
        result = agent.parse_command("Orbit the red cup")
        assert "object" in result
        assert "shot_type" in result
        assert result["shot_type"] == "orbit"
        assert "cup" in result["object"].lower()

    def test_parse_command_follow(self):
        """Test follow command parsing."""
        agent = DirectorAgent(use_llm=False)
        result = agent.parse_command("Follow the cat")
        assert result["shot_type"] == "follow"
        assert "cat" in result["object"].lower()

    def test_parse_command_dolly(self):
        """Test dolly command parsing."""
        agent = DirectorAgent(use_llm=False)
        result = agent.parse_command("Dolly in on the object")
        assert result["shot_type"] == "dolly_in"

    def test_ground_object(self):
        """Test object grounding."""
        agent = DirectorAgent(detection_model=MockDetectionModel(), use_llm=False)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bbox = agent.ground_object(image, "test object")
        assert bbox is not None
        assert bbox.width > 0
        assert bbox.height > 0

    def test_process_command(self):
        """Test full command processing."""
        agent = DirectorAgent(detection_model=MockDetectionModel(), use_llm=False)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = agent.process_command("Follow the object", image)
        assert result is not None
        assert "object_name" in result
        assert "shot_type" in result
        assert "bounding_box" in result
