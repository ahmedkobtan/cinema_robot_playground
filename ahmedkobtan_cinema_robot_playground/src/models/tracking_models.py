"""Tracking model interfaces for object tracking."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from loguru import logger


class TrackingState:
    """State of tracked object."""

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],  # (x, y, width, height)
        confidence: float = 1.0,
        track_id: int = 0,
    ):
        """
        Initialize tracking state.

        Args:
            bbox: Bounding box (x, y, width, height)
            confidence: Tracking confidence (0-1)
            track_id: Unique track ID
        """
        self.bbox = bbox
        self.confidence = confidence
        self.track_id = track_id

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    @property
    def area(self) -> float:
        """Get bounding box area."""
        _, _, w, h = self.bbox
        return w * h


class Tracker(ABC):
    """Base class for object trackers."""

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        initial_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[TrackingState]:
        """
        Update tracker with new frame.

        Args:
            frame: Current video frame
            initial_bbox: Initial bounding box for first frame (x, y, width, height)

        Returns:
            Tracking state or None if tracking lost
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if tracker is available."""
        pass


class BotSORTTracker(Tracker):
    """Bot-SORT tracker for multi-object tracking."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize Bot-SORT tracker.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device
        self.tracker = None
        self._initialized = False
        self._track_id = 0

    def _initialize_tracker(self) -> bool:
        """Initialize Bot-SORT tracker."""
        try:
            import warnings

            import torch  # noqa: F401 - used by BotSort
            from boxmot import BotSort  # Note: BotSort, not BoTSORT

            logger.info("Initializing Bot-SORT tracker")
            import torch as torch_module

            # Bot-SORT expects device as string: "cpu", "0", "1", etc. (not "cuda")
            # Convert "cuda" to "0" (first GPU device)
            device_str = self.device or (
                "cuda" if torch_module.cuda.is_available() else "cpu"
            )
            if device_str == "cuda":
                if torch_module.cuda.is_available():
                    device_str = "0"  # Use first GPU device
                else:
                    device_str = "cpu"  # Fallback to CPU if CUDA not available
            elif device_str.startswith("cuda:"):
                # Handle "cuda:0" format - extract device ID
                device_str = device_str.split(":")[1]

            from pathlib import Path

            # Bot-SORT requires reid_weights as first positional arg
            # Check resources directory first, then current directory

            resources_dir = Path(__file__).parent.parent.parent / "resources"
            reid_weights_path = resources_dir / "osnet_x0_25_msmt17.pt"

            if not reid_weights_path.exists():
                # Fallback to current directory (boxmot will auto-download if needed)
                reid_weights_path = Path("osnet_x0_25_msmt17.pt")

            # Suppress ECC warnings (they're harmless - just mean no camera motion detected)
            # CMC is useful when camera is moving (pan/tilt/wheels), so we keep it enabled
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*ECC did not converge.*")
                self.tracker = BotSort(
                    reid_weights=reid_weights_path,
                    device=device_str,  # Pass as string, not torch.device object
                    half=False,  # Use full precision (half=True for FP16 on GPU)
                    # cmc_method="ecc" is default - keep enabled for moving camera scenarios
                )
            self._initialized = True
            logger.info("Bot-SORT tracker initialized")
            return True

        except ImportError:
            logger.warning("Bot-SORT not available, using fallback")
            return False
        except Exception as e:
            logger.error(f"Error initializing Bot-SORT: {e}")
            return False

    def update(
        self,
        frame: np.ndarray,
        initial_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[TrackingState]:
        """Update tracker with new frame."""
        if not self.is_available():
            if not self._initialize_tracker():
                return None

        try:
            if initial_bbox is not None:
                # Initialize with bounding box
                x, y, w, h = initial_bbox
                detections = np.array(
                    [[x, y, x + w, y + h, 0.9, 0]]
                )  # [x1, y1, x2, y2, conf, class]
                tracks = self.tracker.update(detections, frame)
            else:
                # Continue tracking
                tracks = self.tracker.update(np.array([]), frame)

            if len(tracks) > 0:
                # Get first track
                track = tracks[0]
                x1, y1, x2, y2, track_id, conf = track[:6]
                return TrackingState(
                    bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    confidence=float(conf),
                    track_id=int(track_id),
                )

            return None

        except Exception as e:
            logger.error(f"Error in Bot-SORT tracking: {e}")
            return None

    def reset(self) -> None:
        """Reset tracker."""
        if self.tracker is not None:
            self.tracker.reset()
        self._track_id = 0

    def is_available(self) -> bool:
        """Check if tracker is available."""
        if not self._initialized:
            # Try to initialize if not already done
            return self._initialize_tracker()
        return True and self.tracker is not None


class SimpleTracker(Tracker):
    """Simple template-based tracker for testing."""

    def __init__(self):
        """Initialize simple tracker."""
        self.template = None
        self.bbox = None
        self._available = True

    def update(
        self,
        frame: np.ndarray,
        initial_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[TrackingState]:
        """Update tracker using template matching."""
        if initial_bbox is not None:
            # Initialize
            x, y, w, h = [int(v) for v in initial_bbox]
            self.bbox = (float(x), float(y), float(w), float(h))
            # Extract template
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.template = gray[y : y + h, x : x + w]
            return TrackingState(bbox=self.bbox, confidence=1.0, track_id=0)

        if self.template is None or self.bbox is None:
            return None

        try:
            import cv2

            # Template matching
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.5:  # Confidence threshold
                x, y = max_loc
                w, h = self.template.shape[::-1]
                self.bbox = (float(x), float(y), float(w), float(h))
                return TrackingState(
                    bbox=self.bbox, confidence=float(max_val), track_id=0
                )

            return None

        except Exception as e:
            logger.error(f"Error in simple tracking: {e}")
            return None

    def reset(self) -> None:
        """Reset tracker."""
        self.template = None
        self.bbox = None

    def is_available(self) -> bool:
        """Check if tracker is available."""
        return self._available


class MockTracker(Tracker):
    """Mock tracker for testing."""

    def __init__(self):
        """Initialize mock tracker."""
        self.bbox = None
        self._available = True

    def update(
        self,
        frame: np.ndarray,
        initial_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[TrackingState]:
        """Return mock tracking state."""
        if initial_bbox is not None:
            self.bbox = initial_bbox
        elif self.bbox is None:
            # Initialize with center
            h, w = frame.shape[:2]
            box_size = min(w, h) * 0.2
            x = (w - box_size) / 2
            y = (h - box_size) / 2
            self.bbox = (float(x), float(y), float(box_size), float(box_size))

        return TrackingState(bbox=self.bbox, confidence=0.9, track_id=0)

    def reset(self) -> None:
        """Reset tracker."""
        self.bbox = None

    def is_available(self) -> bool:
        """Mock tracker is always available."""
        return True
