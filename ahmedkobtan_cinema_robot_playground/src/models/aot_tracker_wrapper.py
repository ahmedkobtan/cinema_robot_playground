"""AOT (Associating Objects with Transformers) tracker wrapper.

This wrapper integrates AOT from Segment-and-Track-Anything to provide
mask-based tracking that matches original FAn's approach.

AOT is better than Bot-SORT for mask-based tracking:
- Tracks continuously without periodic re-detections
- More robust to occlusion and deformation
- Simpler logic (just track every frame)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
    Tracker,
    TrackingState,
)


class AOTTrackerWrapper(Tracker):
    """
    Wrapper for AOT tracker from Segment-and-Track-Anything.

    AOT tracks segmentation masks directly, so we need to:
    1. Convert initial bbox to mask (using SAM or simple mask)
    2. Track mask across frames
    3. Convert tracked mask back to bbox for our pipeline
    """

    def __init__(
        self,
        device: Optional[str] = None,
        aot_model: str = "r50_deaotl",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize AOT tracker.

        Args:
            device: Device to use ('cuda' or 'cpu')
            aot_model: AOT model type ('r50_deaotl', 'deaotl', etc.)
            checkpoint_path: Path to AOT checkpoint file
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.aot_model = aot_model
        self.checkpoint_path = checkpoint_path
        self.tracker = None
        self._initialized = False
        self._current_mask = None  # Current tracked mask
        self._track_id = 0

    def _initialize_tracker(self) -> bool:
        """Initialize AOT tracker from Segment-and-Track-Anything."""
        try:
            # Check if FollowAnything directory exists
            follow_anything_dir = (
                Path(__file__).parent.parent.parent.parent / "FollowAnything"
            )
            if not follow_anything_dir.exists():
                logger.warning(
                    "FollowAnything directory not found. AOT tracker requires Segment-and-Track-Anything."
                )
                return False

            # Add Segment-and-Track-Anything to path
            segtrack_dir = follow_anything_dir / "Segment-and-Track-Anything"
            if not segtrack_dir.exists():
                logger.warning("Segment-and-Track-Anything directory not found")
                return False

            # Add to Python path
            if str(segtrack_dir) not in sys.path:
                sys.path.insert(0, str(segtrack_dir))

            # Import AOT tracker
            from aot_tracker import get_aot  # type: ignore

            # AOT model to checkpoint mapping
            aot_model2ckpt = {
                "deaotb": "Segment-and-Track-Anything/ckpt/DeAOTB_PRE_YTB_DAV.pth",
                "deaotl": "Segment-and-Track-Anything/ckpt/DeAOTL_PRE_YTB_DAV.pth",
                "r50_deaotl": "Segment-and-Track-Anything/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
            }

            # Get checkpoint path
            if self.checkpoint_path is None:
                if self.aot_model in aot_model2ckpt:
                    # Use relative path from segtrack_dir
                    checkpoint_rel = aot_model2ckpt[self.aot_model]
                    checkpoint_path = segtrack_dir / checkpoint_rel

                    # Also check resources directory
                    resources_dir = Path(__file__).parent.parent.parent / "resources"
                    checkpoint_name = os.path.basename(checkpoint_path)
                    resources_checkpoint = resources_dir / checkpoint_name

                    if resources_checkpoint.exists():
                        checkpoint_path = resources_checkpoint
                    elif not checkpoint_path.exists():
                        logger.warning(
                            f"AOT checkpoint not found: {checkpoint_path}. "
                            f"Please download from: https://github.com/z-x-yang/Segment-and-Track-Anything"
                        )
                        return False
                else:
                    logger.warning(f"Unknown AOT model: {self.aot_model}")
                    return False
            else:
                checkpoint_path = Path(self.checkpoint_path)
                if not checkpoint_path.exists():
                    logger.warning(f"AOT checkpoint not found: {checkpoint_path}")
                    return False

            # AOT arguments
            gpu_id = 0 if self.device != "cpu" and torch.cuda.is_available() else 0
            if self.device == "cpu":
                logger.warning("AOT tracker works best on GPU. CPU mode may be slow.")

            aot_args = {
                "phase": "PRE_YTB_DAV",
                "model": self.aot_model,
                "model_path": str(checkpoint_path),
                "long_term_mem_gap": 9999,
                "gpu_id": gpu_id,
            }

            # Initialize AOT tracker
            logger.info(f"Initializing AOT tracker ({self.aot_model})...")
            self.tracker = get_aot(aot_args)
            self._initialized = True
            logger.info("AOT tracker initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"AOT tracker not available: {e}")
            logger.info(
                "AOT requires Segment-and-Track-Anything framework. "
                "Make sure FollowAnything/Segment-and-Track-Anything exists."
            )
            return False
        except Exception as e:
            logger.error(f"Error initializing AOT tracker: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _bbox_to_mask(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Convert bounding box to mask.

        Args:
            frame: Image frame
            bbox: Bounding box (x, y, width, height)

        Returns:
            Binary mask (0 or 1)
        """
        h, w = frame.shape[:2]
        x, y, bbox_w, bbox_h = bbox

        # Create mask from bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + bbox_w))
        y2 = min(h, int(y + bbox_h))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

        return mask

    def _mask_to_bbox(
        self, mask: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Convert mask to bounding box.

        Args:
            mask: Binary mask (0 or 1)

        Returns:
            Bounding box (x, y, width, height) or None if mask is empty
        """
        y_indices, x_indices = np.where(mask > 0.5)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        x_min, x_max = float(x_indices.min()), float(x_indices.max())
        y_min, y_max = float(y_indices.min()), float(y_indices.max())

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def update(
        self,
        frame: np.ndarray,
        initial_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[TrackingState]:
        """
        Update tracker with new frame.

        Args:
            frame: Current video frame (BGR format)
            initial_bbox: Initial bounding box for first frame (x, y, width, height)

        Returns:
            Tracking state or None if tracking lost
        """
        if not self.is_available():
            if not self._initialize_tracker():
                return None

        try:
            # Convert BGR to RGB (AOT expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if initial_bbox is not None:
                # Initialize tracking with mask from bbox
                initial_mask = self._bbox_to_mask(frame_rgb, initial_bbox)
                # AOT expects mask as float tensor with shape (1, 1, H, W)
                mask_tensor = (
                    torch.from_numpy(initial_mask).float().unsqueeze(0).unsqueeze(0)
                )

                # Add reference frame
                self.tracker.add_reference_frame(
                    frame_rgb, mask_tensor, obj_nums=1, frame_step=0
                )
                self._current_mask = initial_mask
                self._track_id = 0

                # Return initial state
                return TrackingState(
                    bbox=initial_bbox,
                    confidence=1.0,
                    track_id=self._track_id,
                )

            # Continue tracking
            if self._current_mask is None:
                return None

            # Track mask
            pred_mask_tensor = self.tracker.track(frame_rgb)

            # Update memory for next frame
            self.tracker.update_memory(pred_mask_tensor)

            # Convert mask tensor to numpy
            pred_mask = pred_mask_tensor.squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

            # Check if mask is empty (tracking lost)
            if np.sum(pred_mask) == 0:
                self._current_mask = None
                return None

            self._current_mask = pred_mask

            # Convert mask to bbox
            bbox = self._mask_to_bbox(pred_mask)
            if bbox is None:
                return None

            # Calculate confidence based on mask stability
            # AOT doesn't provide confidence, so we use mask area as proxy
            mask_area = np.sum(pred_mask)
            h, w = frame.shape[:2]
            image_area = h * w
            confidence = min(1.0, mask_area / (image_area * 0.1))  # Normalize

            return TrackingState(
                bbox=bbox,
                confidence=float(confidence),
                track_id=self._track_id,
            )

        except Exception as e:
            logger.error(f"Error in AOT tracking: {e}")
            import traceback

            traceback.print_exc()
            return None

    def reset(self) -> None:
        """Reset tracker."""
        if self.tracker is not None:
            self.tracker.restart()
        self._current_mask = None
        self._track_id = 0

    def is_available(self) -> bool:
        """Check if tracker is available."""
        if not self._initialized:
            return self._initialize_tracker()
        return self.tracker is not None

    def get_mask(self) -> Optional[np.ndarray]:
        """
        Get current tracked mask (for visualization or further processing).

        Returns:
            Current mask or None if not tracking
        """
        return self._current_mask
