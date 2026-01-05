"""Detection model interfaces for object grounding and detection."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger


class BoundingBox:
    """Bounding box representation."""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        confidence: float = 1.0,
        class_name: Optional[str] = None,
    ):
        """
        Initialize bounding box.

        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Box width
            height: Box height
            confidence: Detection confidence (0-1)
            class_name: Optional class name/label for the detected object
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.class_name = class_name

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        """Get box area."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }
        if self.class_name is not None:
            result["class_name"] = self.class_name
        return result


class DetectionModel(ABC):
    """Base class for object detection models."""

    @abstractmethod
    def detect(
        self, image: np.ndarray, text_prompt: Optional[str] = None
    ) -> List[BoundingBox]:
        """
        Detect objects in image.

        Args:
            image: Input image (BGR format)
            text_prompt: Optional text prompt for open-vocabulary detection

        Returns:
            List of bounding boxes
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available (loaded and ready)."""
        pass


class GroundingDINOModel(DetectionModel):
    """Grounding DINO model for open-vocabulary object detection."""

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None,
    ):
        """
        Initialize Grounding DINO model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False

    def _load_model(self) -> bool:
        """Load model from HuggingFace."""
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info(f"Loading Grounding DINO model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name
            ).to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("Grounding DINO model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading Grounding DINO: {e}")
            return False

    def detect(
        self, image: np.ndarray, text_prompt: Optional[str] = None
    ) -> List[BoundingBox]:
        """Detect objects using Grounding DINO."""
        if not self.is_available():
            if not self._load_model():
                return []

        if text_prompt is None:
            logger.warning("No text prompt provided for Grounding DINO")
            return []

        try:
            import cv2

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image and text
            inputs = self.processor(
                images=image_rgb, text=text_prompt, return_tensors="pt"
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Process results with adaptive threshold
            # Lower threshold for better recall, especially for objects like "lamp"
            # Can filter by confidence later if needed
            threshold = (
                0.15  # Lowered from 0.18 for better detection of difficult objects
            )

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=[(image.shape[0], image.shape[1])],
                threshold=threshold,
            )[0]

            boxes = []
            h, w = image.shape[:2]
            image_area = h * w

            for box, score, label in zip(
                results["boxes"],
                results["scores"],
                results["labels"],
            ):
                # Filter false positives by checking if label is semantically similar to prompt
                # Use a general approach: check if any significant words from prompt appear in label
                # This helps filter obvious mismatches without hard-coding specific objects
                label_str = str(label).lower() if label else ""
                prompt_lower = text_prompt.lower() if text_prompt else ""

                if text_prompt and label_str:
                    # Extract meaningful words from prompt (ignore common words)
                    common_words = {
                        "a",
                        "an",
                        "the",
                        "is",
                        "are",
                        "was",
                        "were",
                        "of",
                        "in",
                        "on",
                        "at",
                        "to",
                        "for",
                    }
                    prompt_words = {
                        w
                        for w in prompt_lower.split()
                        if w not in common_words and len(w) > 2
                    }

                    # If prompt has meaningful words, check if label contains any of them
                    # This is a general similarity check, not object-specific
                    if prompt_words:
                        # Check if any prompt word appears in label (basic semantic similarity)
                        has_similarity = any(word in label_str for word in prompt_words)

                        # If no similarity at all and confidence is low, might be a false positive
                        # But don't reject based on this alone - let confidence score handle it
                        # Only log for debugging
                        if not has_similarity and score < 0.3:
                            logger.debug(
                                f"Low confidence detection with no semantic similarity: "
                                f"prompt '{text_prompt}' vs label '{label_str}' (score: {score:.2f})"
                            )

                x, y, x2, y2 = box.cpu().numpy()
                bbox_width = float(x2 - x)
                bbox_height = float(y2 - y)
                bbox_area = bbox_width * bbox_height
                area_coverage = bbox_area / image_area if image_area > 0 else 0

                # Reject full-frame detections (cover >95% of image)
                if area_coverage < 0.95:
                    boxes.append(
                        BoundingBox(
                            x=float(x),
                            y=float(y),
                            width=bbox_width,
                            height=bbox_height,
                            confidence=float(score),
                        )
                    )

            return boxes

        except Exception as e:
            logger.error(f"Error in Grounding DINO detection: {e}")
            return []

    def is_available(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self.model is not None


class MockDetectionModel(DetectionModel):
    """Mock detection model for testing without GPU."""

    def __init__(self):
        """Initialize mock model."""
        self._available = True

    def detect(
        self, image: np.ndarray, text_prompt: Optional[str] = None
    ) -> List[BoundingBox]:
        """Return mock detection (center of frame)."""
        h, w = image.shape[:2]
        # Return a box in the center
        box_size = min(w, h) * 0.2
        x = (w - box_size) / 2
        y = (h - box_size) / 2

        return [
            BoundingBox(
                x=float(x),
                y=float(y),
                width=float(box_size),
                height=float(box_size),
                confidence=0.9,
            )
        ]

    def is_available(self) -> bool:
        """Mock model is always available."""
        return True
