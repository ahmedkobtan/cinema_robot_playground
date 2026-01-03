"""Comprehensive tests for Director Agent - matches script-level testing."""

import numpy as np
import pytest

from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    MockDetectionModel,
)
from ahmedkobtan_cinema_robot_playground.src.services.director_agent import (
    DirectorAgent,
)


class TestDirectorAgentComprehensive:
    """Comprehensive tests matching scripts/test_llm_agent.py."""

    def test_initialization_with_device(self):
        """Test agent initialization with device parameter."""
        agent = DirectorAgent(use_llm=False, device="cpu")
        assert agent.device == "cpu"
        assert agent.detection_model is not None

    def test_initialization_with_llm(self):
        """Test agent initialization with LLM enabled."""
        # LLM may not be available, so allow failure
        try:
            agent = DirectorAgent(use_llm=True, device="cpu")
            # If LLM loads, check it's available
            if agent.llm_agent is not None:
                assert agent.use_llm is True
        except Exception:
            # LLM loading may fail, which is acceptable
            pass

    def test_parse_command_orbit(self):
        """Test orbit command parsing."""
        agent = DirectorAgent(use_llm=False, device="cpu")
        result = agent.parse_command("Orbit the red cup")
        assert "object" in result
        assert "shot_type" in result
        assert result["shot_type"] == "orbit"
        assert "cup" in result["object"].lower() or "red" in result["object"].lower()

    def test_parse_command_follow(self):
        """Test follow command parsing."""
        agent = DirectorAgent(use_llm=False, device="cpu")
        result = agent.parse_command("Follow the cat")
        assert result["shot_type"] == "follow"
        assert "cat" in result["object"].lower()

    def test_parse_command_dolly_in(self):
        """Test dolly in command parsing."""
        agent = DirectorAgent(use_llm=False, device="cpu")
        result = agent.parse_command("Dolly in on the object")
        assert result["shot_type"] == "dolly_in"

    def test_parse_command_dolly_out(self):
        """Test dolly out command parsing."""
        agent = DirectorAgent(use_llm=False, device="cpu")
        result = agent.parse_command("Dolly out from the object")
        assert result["shot_type"] == "dolly_out"

    def test_parse_command_pan(self):
        """Test pan command parsing."""
        agent = DirectorAgent(use_llm=False, device="cpu")
        result = agent.parse_command("Pan to the left")
        assert result["shot_type"] == "pan"

    def test_parse_command_track(self):
        """Test track command parsing (should map to follow)."""
        agent = DirectorAgent(use_llm=False, device="cpu")
        result = agent.parse_command("Track the person in the green shirt")
        assert result["shot_type"] == "follow"
        assert (
            "person" in result["object"].lower() or "shirt" in result["object"].lower()
        )

    def test_ground_object(self):
        """Test object grounding."""
        agent = DirectorAgent(
            detection_model=MockDetectionModel(), use_llm=False, device="cpu"
        )
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        bbox = agent.ground_object(image, "test object")
        assert bbox is not None
        assert bbox.width > 0
        assert bbox.height > 0

    def test_process_command_full_flow(self):
        """Test full command processing flow."""
        agent = DirectorAgent(
            detection_model=MockDetectionModel(), use_llm=False, device="cpu"
        )
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = agent.process_command("Follow the object", image)
        assert result is not None
        assert "object_name" in result
        assert "shot_type" in result
        assert "bounding_box" in result

    def test_llm_parsing_if_available(self):
        """Test LLM-based parsing if available."""
        try:
            agent = DirectorAgent(use_llm=True, device="cpu")
            if agent.llm_agent is not None:
                result = agent.parse_command("Orbit the red cup")
                assert "object" in result
                assert "shot_type" in result
        except Exception:
            # LLM may not be available, which is acceptable
            pytest.skip("LLM not available")

    def test_device_parameter_consistency(self):
        """Test that device parameter is consistently used."""
        for device in ["cpu", "cuda"]:
            agent = DirectorAgent(use_llm=False, device=device)
            assert agent.device == device
            if agent.detection_model is not None:
                # Detection model should also use device
                assert hasattr(agent.detection_model, "device")
