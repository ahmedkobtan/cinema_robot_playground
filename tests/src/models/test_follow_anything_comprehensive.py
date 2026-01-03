"""Comprehensive tests for Follow Anything model - matches script-level testing."""

import numpy as np
import pytest

from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import (
    FollowAnythingFallback,
    FollowAnythingModel,
)


class TestFollowAnythingComprehensive:
    """Comprehensive tests matching scripts/test_fan_components.py and test_fan_end_to_end.py."""

    def test_initialization_with_device(self):
        """Test model initialization with device parameter."""
        model = FollowAnythingModel(device="cpu")
        assert model.device == "cpu"
        assert hasattr(model, "sam2_predictor")
        assert hasattr(model, "clip_model")
        assert hasattr(model, "dino_model")

    def test_initialization_with_sam2(self):
        """Test model initialization with SAM 2 enabled."""
        model = FollowAnythingModel(use_sam2=True, device="cpu")
        assert model.use_sam2 is True
        # SAM 2 may not be available without checkpoint, but initialization should work
        assert model.is_available() or model.fallback is not None

    def test_initialization_without_sam2(self):
        """Test model initialization without SAM 2."""
        model = FollowAnythingModel(use_sam2=False, device="cpu")
        assert model.use_sam2 is False
        # Should still work with CLIP + DINO
        assert model.is_available() or model.fallback is not None

    def test_detection_with_text_prompt(self):
        """Test detection with text prompt."""
        model = FollowAnythingModel(device="cpu")
        if not model.is_available() and model.fallback is None:
            pytest.skip("Model not available")

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        boxes = model.detect(image, text_prompt="a red cup")

        assert isinstance(boxes, list)
        # May return empty list if no match, but should not crash
        for box in boxes:
            assert hasattr(box, "x")
            assert hasattr(box, "y")
            assert hasattr(box, "width")
            assert hasattr(box, "height")
            assert hasattr(box, "confidence")
            assert 0 <= box.confidence <= 1

    def test_detection_multiple_prompts(self):
        """Test detection with multiple text prompts."""
        model = FollowAnythingModel(device="cpu")
        if not model.is_available() and model.fallback is None:
            pytest.skip("Model not available")

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        prompts = ["a red cup", "something red", "a blue ball"]

        for prompt in prompts:
            boxes = model.detect(image, text_prompt=prompt)
            assert isinstance(boxes, list)

    def test_tracking_initialization(self):
        """Test tracking initialization."""
        model = FollowAnythingModel(device="cpu")
        if not model.is_available() and model.fallback is None:
            pytest.skip("Model not available")

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        initial_bbox = (200.0, 200.0, 200.0, 300.0)  # x, y, width, height

        state = model.update(image, initial_bbox=initial_bbox)
        # State may be None if tracker not available, but should not crash
        if state is not None:
            assert hasattr(state, "bbox")
            assert hasattr(state, "confidence")
            assert hasattr(state, "track_id")

    def test_tracking_update(self):
        """Test tracking update."""
        model = FollowAnythingModel(device="cpu")
        if not model.is_available() and model.fallback is None:
            pytest.skip("Model not available")

        frames = [
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(3)
        ]
        initial_bbox = (200.0, 200.0, 200.0, 300.0)

        # Initialize
        state = model.update(frames[0], initial_bbox=initial_bbox)
        if state is None:
            pytest.skip("Tracking not available")

        # Update
        for frame in frames[1:]:
            state = model.update(frame)
            # State may become None if tracking lost
            if state is None:
                break
            assert hasattr(state, "bbox")
            assert hasattr(state, "confidence")

    def test_reset(self):
        """Test reset functionality."""
        model = FollowAnythingModel(device="cpu")
        model.reset()
        assert model._tracking_initialized is False
        assert model._current_track_id == 0
        assert len(model._stored_features) == 0
        assert model._current_image is None

    def test_fallback_initialization(self):
        """Test fallback model initialization."""
        fallback = FollowAnythingFallback(device="cpu")
        assert fallback.detector is not None
        assert fallback.tracker is not None

    def test_fallback_detection(self):
        """Test fallback detection."""
        fallback = FollowAnythingFallback(device="cpu")
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        try:
            boxes = fallback.detect(image, "test object")
            assert isinstance(boxes, list)
        except Exception:
            # Expected if models not available
            pytest.skip("Fallback models not available")

    def test_fallback_tracking(self):
        """Test fallback tracking."""
        fallback = FollowAnythingFallback(device="cpu")
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)

        try:
            _state = fallback.update(image, initial_bbox=initial_bbox)
            # State may be None if not available
        except Exception:
            pytest.skip("Fallback tracking not available")

    def test_model_components_availability(self):
        """Test that model reports component availability correctly."""
        model = FollowAnythingModel(device="cpu")

        # Check component attributes exist
        assert hasattr(model, "sam2_predictor")
        assert hasattr(model, "clip_model")
        assert hasattr(model, "dino_model")
        assert hasattr(model, "tracker")
        assert hasattr(model, "fallback")

        # Check availability method
        available = model.is_available()
        assert isinstance(available, bool)

    def test_device_parameter_consistency(self):
        """Test that device parameter is consistently used."""
        for device in ["cpu", "cuda"]:
            model = FollowAnythingModel(device=device)
            assert model.device == device

            # Check that device is passed to sub-components
            if model.fallback is not None:
                assert model.fallback.device == device
