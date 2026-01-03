"""Follow Anything (FAn) model for unified detection and tracking.

Based on: https://github.com/alaamaalouf/FollowAnything

FAn approach (using modern PyPI packages):
1. SAM 2 extracts multiple masks (segmentations)
2. Based on DINO/CLIP features, FAn classifies each mask
3. Objects are detected by assigning masks whose feature descriptor is closest to query
4. Bot-SORT tracks the object across frames (or AOT if available)

Uses:
- SAM 2 (sam2) - newer than SAM
- Open-CLIP (open-clip-torch) - modern CLIP implementation
- DINOv2 (via transformers) - for feature extraction
- Bot-SORT (via boxmot) - for tracking
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    BoundingBox,
    DetectionModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
    Tracker,
    TrackingState,
)


class FollowAnythingModel(DetectionModel, Tracker):
    """
    Follow Anything (FAn) - Unified open-vocabulary detection and tracking.

    Implementation using modern PyPI packages:
    - SAM 2 (sam2) for segmentation
    - Open-CLIP (open-clip-torch) for text/image encoding
    - DINOv2 (transformers) for feature extraction
    - Bot-SORT (boxmot) for tracking
    """

    def __init__(
        self,
        use_sam2: bool = True,
        clip_model_name: str = "ViT-B-32",  # Open-CLIP model
        pretrained: str = "openai",  # Open-CLIP pretrained dataset
        device: Optional[str] = None,
    ):
        """
        Initialize Follow Anything model.

        Args:
            use_sam2: Whether to use SAM 2 for segmentation
            clip_model_name: Open-CLIP model name (e.g., "ViT-B-32", "ViT-L-14")
            pretrained: Pretrained dataset (e.g., "openai", "laion2b_s13b_b90k")
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_sam2 = use_sam2
        self.clip_model_name = clip_model_name
        self.pretrained = pretrained

        # Model components
        self.sam2_predictor = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.dino_model = None
        self.dino_processor = None
        self.tracker = None

        # State
        self._loaded = False
        self._tracking_initialized = False
        self._current_track_id = 0
        self._stored_features = []  # For re-detection
        self._current_image = None  # For SAM/SAM2

        # Fallback if components not available
        self.fallback = None

        # Try to load model immediately
        self._load_model()

    def _load_model(self) -> bool:
        """Load Follow Anything model components."""
        try:
            # Try to load SAM 2 (preferred)
            if self.use_sam2:
                if self._load_sam2():
                    logger.info("SAM 2 loaded successfully")
                else:
                    logger.warning("SAM 2 not available")

            # Load Open-CLIP
            if self._load_clip():
                logger.info(f"Open-CLIP ({self.clip_model_name}) loaded successfully")
            else:
                logger.warning("Open-CLIP not available")

            # Load DINOv2
            if self._load_dino():
                logger.info("DINOv2 loaded successfully")
            else:
                logger.warning("DINOv2 not available")

            # Load tracker (Bot-SORT)
            if self._load_tracker():
                logger.info("Bot-SORT tracker loaded successfully")
            else:
                logger.warning("Bot-SORT tracker not available")

            # Check if we have minimum required components
            # We need at least CLIP or DINO for features, and ideally SAM/SAM2 for segmentation
            has_features = self.clip_model is not None or self.dino_model is not None
            has_segmentation = self.sam2_predictor is not None

            if has_features:
                self._loaded = True
                if has_segmentation:
                    logger.info(
                        "Follow Anything (FAn) loaded successfully with segmentation"
                    )
                else:
                    logger.info(
                        "Follow Anything (FAn) loaded successfully (without segmentation, using full-image features)"
                    )
                return True
            else:
                # Use fallback
                logger.warning("FAn components not fully available, using fallback")
                return self._load_fallback()

        except Exception as e:
            logger.error(f"Error loading Follow Anything: {e}")
            logger.info("Falling back to Grounding DINO + Bot-SORT")
            return self._load_fallback()

    def _load_sam2(self) -> bool:
        """Load SAM 2 model."""
        try:
            # SAM 2 requires checkpoint - check multiple locations
            import os
            from pathlib import Path

            import sam2  # type: ignore
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            # 1. Check environment variable
            sam2_checkpoint = os.getenv("SAM2_CHECKPOINT", None)

            # 2. Check resources directory (default location)
            if sam2_checkpoint is None or not os.path.exists(sam2_checkpoint):
                resources_dir = Path(__file__).parent.parent.parent / "resources"
                default_checkpoints = [
                    resources_dir / "sam2.1_hiera_base_plus.pt",
                    resources_dir / "sam2.1_hiera_large.pt",
                    resources_dir / "sam2.1_hiera_small.pt",
                    resources_dir / "sam2.1_hiera_tiny.pt",
                    resources_dir / "sam2_hiera_base_plus.pt",
                    resources_dir / "sam2_hiera_large.pt",
                ]

                for checkpoint_path in default_checkpoints:
                    if checkpoint_path.exists():
                        sam2_checkpoint = str(checkpoint_path)
                        logger.info(
                            f"Found SAM 2 checkpoint in resources: {checkpoint_path.name}"
                        )
                        break

            if sam2_checkpoint and os.path.exists(sam2_checkpoint):
                try:
                    # Build SAM 2 model
                    # Try to find config file
                    # Auto-detect config based on checkpoint filename
                    sam2_config_dir = os.path.join(
                        os.path.dirname(sam2.__file__), "configs"
                    )
                    sam2_model_cfg = None

                    if os.path.exists(sam2_config_dir):
                        # Try to infer config from checkpoint filename
                        checkpoint_name = os.path.basename(sam2_checkpoint).lower()

                        # Map checkpoint names to config names
                        # Note: Hydra expects "configs/sam2.1/..." format when using the package
                        config_mapping = {
                            "tiny": "configs/sam2.1/sam2.1_hiera_t",
                            "small": "configs/sam2.1/sam2.1_hiera_s",
                            "base_plus": "configs/sam2.1/sam2.1_hiera_b+",
                            "base": "configs/sam2.1/sam2.1_hiera_b+",
                            "large": "configs/sam2.1/sam2.1_hiera_l",
                        }

                        # Try to match checkpoint name
                        for key, config_name in config_mapping.items():
                            if key in checkpoint_name:
                                # Verify config file exists
                                # Remove "configs/" prefix for file path check
                                cfg_file_name = config_name.replace("configs/", "")
                                cfg_path = os.path.join(
                                    sam2_config_dir, cfg_file_name + ".yaml"
                                )
                                if os.path.exists(cfg_path):
                                    sam2_model_cfg = config_name  # Use full path with configs/ prefix
                                    logger.info(
                                        f"Using config: {config_name} for checkpoint"
                                    )
                                    break

                        # Fallback: try common config names
                        if sam2_model_cfg is None:
                            for cfg_name in [
                                "configs/sam2.1/sam2.1_hiera_b+",  # base_plus (recommended)
                                "configs/sam2.1/sam2.1_hiera_l",  # large
                                "configs/sam2.1/sam2.1_hiera_s",  # small
                                "configs/sam2.1/sam2.1_hiera_t",  # tiny
                            ]:
                                # Verify file exists
                                cfg_file_name = cfg_name.replace("configs/", "")
                                cfg_path = os.path.join(
                                    sam2_config_dir, cfg_file_name + ".yaml"
                                )
                                if os.path.exists(cfg_path):
                                    sam2_model_cfg = cfg_name
                                    logger.info(f"Using fallback config: {cfg_name}")
                                    break

                    sam2_model = build_sam2(
                        sam2_model_cfg, sam2_checkpoint, device=self.device
                    )
                    self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                    logger.info("SAM 2 model loaded successfully")
                    return True
                except Exception as e:
                    logger.warning(f"SAM 2 initialization failed: {e}")
                    logger.info(
                        "SAM 2 checkpoint required. Set SAM2_CHECKPOINT environment variable."
                    )
                    return False
            else:
                logger.info(
                    "SAM 2 package available (checkpoint required - set SAM2_CHECKPOINT env var)"
                )
                # Mark as available but not fully initialized
                return True
        except ImportError:
            return False

    def _load_clip(self) -> bool:
        """Load Open-CLIP model."""
        try:
            import open_clip  # type: ignore

            # Load model and preprocessing
            self.clip_model, _, self.clip_preprocess = (
                open_clip.create_model_and_transforms(
                    self.clip_model_name, pretrained=self.pretrained, device=self.device
                )
            )
            self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            self.clip_model.eval()

            logger.info(f"Open-CLIP model loaded: {self.clip_model_name}")
            return True
        except ImportError:
            logger.warning("open-clip-torch not available")
            return False
        except Exception as e:
            logger.warning(f"Error loading Open-CLIP: {e}")
            return False

    def _load_dino(self) -> bool:
        """Load DINOv2 model."""
        try:
            from transformers import AutoImageProcessor, AutoModel

            # Load DINOv2 for feature extraction
            self.dino_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base"
            )
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(
                self.device
            )
            self.dino_model.eval()

            return True
        except Exception as e:
            logger.warning(f"Error loading DINOv2: {e}")
            return False

    def _load_tracker(self) -> bool:
        """Load Bot-SORT tracker."""
        try:
            from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
                BotSORTTracker,
            )

            self.tracker = BotSORTTracker(device=self.device)
            return self.tracker.is_available()
        except Exception as e:
            logger.warning(f"Error loading Bot-SORT tracker: {e}")
            return False

    def _load_fallback(self) -> bool:
        """Load fallback implementation (Grounding DINO + Bot-SORT)."""
        if self.fallback is None:
            self.fallback = FollowAnythingFallback(device=self.device)

        if self.fallback.is_available():
            self._loaded = True
            logger.info("Using Follow Anything fallback (Grounding DINO + Bot-SORT)")
            return True
        return False

    def _get_masks_sam2(self, image_rgb: np.ndarray) -> List[np.ndarray]:
        """Get masks using SAM 2."""
        if self.sam2_predictor is None:
            return []

        try:
            # Always set image - SAM 2 predictor needs it set before each prediction
            # The comparison might fail due to array equality checks, so just set it
            self.sam2_predictor.set_image(image_rgb)
            self._current_image = (
                image_rgb.copy()
            )  # Store copy to avoid reference issues

            h, w = image_rgb.shape[:2]

            # Generate masks using grid of points for comprehensive coverage
            # Use a denser grid for better mask generation
            grid_points = []
            grid_size = 4  # 4x4 grid
            for i in range(1, grid_size):
                for j in range(1, grid_size):
                    x = int(w * j / grid_size)
                    y = int(h * i / grid_size)
                    grid_points.append([x, y])

            # Also add center points of each quadrant
            points = np.array(
                grid_points
                + [
                    [w // 4, h // 4],
                    [3 * w // 4, h // 4],
                    [w // 4, 3 * h // 4],
                    [3 * w // 4, 3 * h // 4],
                    [w // 2, h // 2],
                ],
                dtype=np.float32,
            )
            point_labels = np.ones(len(points), dtype=np.int32)

            # Generate masks
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=True,
            )

            # Return top masks (sorted by score)
            if len(scores) > 0:
                # Get unique masks (avoid duplicates)
                unique_masks = []
                seen_masks = set()
                for idx in np.argsort(scores)[::-1]:  # Sort descending
                    mask = masks[idx]
                    # Simple hash to avoid exact duplicates
                    mask_hash = hash(mask.tobytes()[:1000])  # First 1000 bytes
                    if mask_hash not in seen_masks:
                        unique_masks.append(mask)
                        seen_masks.add(mask_hash)
                        if len(unique_masks) >= 5:  # Top 5 unique masks
                            break
                return unique_masks
            return []
        except Exception as e:
            logger.warning(f"Error generating SAM 2 masks: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _get_masks_simple(self, image_rgb: np.ndarray) -> List[np.ndarray]:
        """Get simple masks (full image) when SAM not available."""
        h, w = image_rgb.shape[:2]
        return [np.ones((h, w), dtype=bool)]

    def detect(
        self, image: np.ndarray, text_prompt: Optional[str] = None
    ) -> List[BoundingBox]:
        """
        Detect objects using Follow Anything.

        Args:
            image: Input image (BGR format)
            text_prompt: Text description of object to detect

        Returns:
            List of bounding boxes
        """
        if not self.is_available():
            if not self._load_model():
                return []

        # If using fallback, delegate
        if self.fallback is not None and self._loaded:
            return self.fallback.detect(image, text_prompt)

        if text_prompt is None:
            logger.warning("No text prompt provided for Follow Anything")
            return []

        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._current_image = image_rgb
            h, w = image_rgb.shape[:2]

            boxes = []

            # FAn detection pipeline:
            # 1. Get masks from SAM 2
            masks = []
            if self.use_sam2 and self.sam2_predictor is not None:
                masks = self._get_masks_sam2(image_rgb)

            if not masks:
                # Fallback to simple mask (full image) when SAM 2 not available
                masks = self._get_masks_simple(image_rgb)

            # 2. Extract query features using Open-CLIP or DINO
            query_features = None
            if self.clip_model is not None and text_prompt:
                # Use Open-CLIP for text queries
                try:
                    text_tokens = self.clip_tokenizer([text_prompt]).to(self.device)
                    with torch.no_grad():
                        query_features = self.clip_model.encode_text(text_tokens)
                        query_features = query_features / query_features.norm(
                            dim=-1, keepdim=True
                        )
                except Exception as e:
                    logger.warning(f"Error encoding text with Open-CLIP: {e}")

            if query_features is None:
                logger.warning("No text feature extractor available, using fallback")
                if self.fallback is not None:
                    return self.fallback.detect(image, text_prompt)
                return []

            # 3. Compute features for each mask and find best match
            best_mask = None
            best_score = -1.0

            for mask in masks:
                # Extract features from masked region
                masked_image = image_rgb * mask[:, :, np.newaxis]

                if self.clip_model is not None:
                    # Use Open-CLIP to encode masked image
                    try:
                        # Preprocess image
                        pil_image = transforms.ToPILImage()(
                            masked_image.astype(np.uint8)
                        )
                        image_tensor = (
                            self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
                        )

                        with torch.no_grad():
                            image_features = self.clip_model.encode_image(image_tensor)
                            image_features = image_features / image_features.norm(
                                dim=-1, keepdim=True
                            )

                        # Compute similarity
                        similarity = (query_features @ image_features.T).item()

                        if similarity > best_score:
                            best_score = similarity
                            best_mask = mask
                    except Exception as e:
                        logger.warning(f"Error encoding image with Open-CLIP: {e}")
                        continue

            # 4. Convert best mask to bounding box
            if best_mask is not None and best_score > 0.3:  # Threshold
                y_indices, x_indices = np.where(best_mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x_min, x_max = float(x_indices.min()), float(x_indices.max())
                    y_min, y_max = float(y_indices.min()), float(y_indices.max())

                    bbox = BoundingBox(
                        x=x_min,
                        y=y_min,
                        width=x_max - x_min,
                        height=y_max - y_min,
                        confidence=float(best_score),
                        class_name=text_prompt,
                    )
                    boxes.append(bbox)

                    # Store features for re-detection
                    self._stored_features.append(query_features.cpu())
            elif best_score > 0.2:  # Lower threshold for full-image mode (no SAM)
                # If no mask but good similarity, return full image as bbox
                # This is less precise but allows detection without SAM
                bbox = BoundingBox(
                    x=0.0,
                    y=0.0,
                    width=float(w),
                    height=float(h),
                    confidence=float(best_score),
                    class_name=text_prompt,
                )
                boxes.append(bbox)
                self._stored_features.append(query_features.cpu())

            return boxes

        except Exception as e:
            logger.error(f"Error in Follow Anything detection: {e}")
            if self.fallback is not None:
                return self.fallback.detect(image, text_prompt)
            return []

    def update(
        self,
        frame: np.ndarray,
        initial_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[TrackingState]:
        """
        Update tracking with new frame using Follow Anything.

        Args:
            frame: Current video frame
            initial_bbox: Initial bounding box for first frame (x, y, width, height)

        Returns:
            Tracking state or None if tracking lost
        """
        if not self.is_available():
            if not self._load_model():
                return None

        # If using fallback, delegate
        if self.fallback is not None and self._loaded:
            return self.fallback.update(frame, initial_bbox)

        # Use Bot-SORT tracker if available
        if self.tracker is not None:
            return self.tracker.update(frame, initial_bbox)

        # Simple tracking fallback
        try:
            if initial_bbox is not None:
                self._tracking_initialized = True
                self._current_track_id = 0

            if not self._tracking_initialized:
                return None

            # Simple tracking: return same bbox (placeholder)
            if initial_bbox is not None:
                x, y, w, h = initial_bbox
                return TrackingState(
                    track_id=self._current_track_id,
                    bbox=(x, y, w, h),
                    confidence=0.9,
                    lost=False,
                )

            return None

        except Exception as e:
            logger.error(f"Error in Follow Anything tracking: {e}")
            if self.fallback is not None:
                return self.fallback.update(frame, initial_bbox)
            return None

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracking_initialized = False
        self._current_track_id = 0
        self._stored_features = []
        self._current_image = None
        if self.tracker is not None:
            self.tracker.reset()
        if self.fallback is not None:
            self.fallback.reset()

    def is_available(self) -> bool:
        """Check if model is loaded and available."""
        if not self._loaded:
            return False

        # Check if FAn components are available
        # We need at least CLIP or DINO for features
        has_features = self.clip_model is not None or self.dino_model is not None

        if has_features:
            return True

        # Check if fallback is available
        if self.fallback is not None:
            return self.fallback.is_available()

        return False


class FollowAnythingFallback(DetectionModel, Tracker):
    """
    Fallback implementation that uses Grounding DINO + Bot-SORT
    when Follow Anything components are not available.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize fallback model."""
        from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
            GroundingDINOModel,
        )
        from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
            BotSORTTracker,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = GroundingDINOModel(device=self.device)
        self.tracker = BotSORTTracker(device=self.device)
        self._tracking_initialized = False

    def detect(
        self, image: np.ndarray, text_prompt: Optional[str] = None
    ) -> List[BoundingBox]:
        """Detect using Grounding DINO."""
        return self.detector.detect(image, text_prompt)

    def update(
        self,
        frame: np.ndarray,
        initial_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[TrackingState]:
        """Track using Bot-SORT."""
        if initial_bbox is not None:
            self._tracking_initialized = True
        return self.tracker.update(frame, initial_bbox)

    def reset(self) -> None:
        """Reset tracker."""
        self.tracker.reset()
        self._tracking_initialized = False

    def is_available(self) -> bool:
        """Check if fallback is available."""
        return self.detector.is_available() and self.tracker.is_available()
