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

from typing import Any, Dict, List, Optional, Tuple

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
        self._stored_features = []  # For re-detection: stored DINO features of tracked object
        self._stored_query_features = None  # Store query features (text prompt)
        self._current_image = None  # For SAM/SAM2
        self._last_tracked_bbox = None  # Last successfully tracked bounding box

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
        """Load Bot-SORT tracker wrapped with SmartTracker for automatic re-detection."""
        try:
            from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
                BotSORTTracker,
                SmartTracker,
            )

            base_tracker = BotSORTTracker(device=self.device)
            if not base_tracker.is_available():
                return False

            # Wrap with SmartTracker that handles re-detection logic
            # Detection callback will use self.detect() for re-detection
            def detection_callback(frame: np.ndarray):
                """Callback for automatic re-detection."""
                if self._stored_query_features is None:
                    return None
                # Use stored text prompt for re-detection
                # We'll need to store the text prompt - for now, return None
                # (re-detection will be handled manually via update() with initial_bbox)
                return None

            self.tracker = SmartTracker(
                base_tracker=base_tracker,
                detection_callback=None,  # Manual re-detection via update() with initial_bbox
                min_hits_to_confirm=5,
                redetect_interval=10,
            )
            return True
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

    def _get_masks_sam2(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get masks using SAM 2, matching original FAn format.

        Original FAn uses sam.seg() which returns list of dicts with keys:
        - 'segmentation': bool mask
        - 'bbox': [x, y, w, h] bounding box
        - 'area': mask area
        - 'point_coords': point used to generate mask

        Returns:
            List of mask dictionaries matching original FAn format.
        """
        if self.sam2_predictor is None:
            return []

        try:
            # Always set image - SAM 2 predictor needs it set before each prediction
            self.sam2_predictor.set_image(image_rgb)
            self._current_image = image_rgb.copy()

            h, w = image_rgb.shape[:2]
            image_area = h * w

            # Original FAn uses automatic mask generation (sam.seg())
            # SAM 2 doesn't have SamAutomaticMaskGenerator, so we simulate it
            # by generating masks from a grid of points (similar to how SAM's
            # automatic mask generator works internally)

            # Generate masks from a grid of points across the image
            # This simulates automatic mask generation
            points_per_side = 16  # Original FAn default (from model_args.py)
            step = min(w, h) // points_per_side

            all_mask_dicts = []
            seen_masks = set()

            # Sample points in a grid pattern
            for y in range(step // 2, h, step):
                for x in range(step // 2, w, step):
                    points = np.array([[x, y]], dtype=np.float32)
                    point_labels = np.array([1], dtype=np.int32)

                    # Generate multiple masks from this point
                    masks, scores, _ = self.sam2_predictor.predict(
                        point_coords=points,
                        point_labels=point_labels,
                        multimask_output=True,
                    )

                    # Convert to FAn format (list of dicts)
                    for mask, score in zip(masks, scores):
                        mask_area = np.sum(mask)
                        coverage = mask_area / image_area if image_area > 0 else 0

                        # Skip full-frame masks (coverage > 90%)
                        if coverage >= 0.90:
                            continue

                        # Skip very small masks
                        if mask_area < 200:  # min_area_size default
                            continue

                        # Get bounding box
                        y_indices, x_indices = np.where(mask)
                        if len(x_indices) == 0 or len(y_indices) == 0:
                            continue

                        x_min, x_max = int(x_indices.min()), int(x_indices.max())
                        y_min, y_max = int(y_indices.min()), int(y_indices.max())
                        bbox = [
                            x_min,
                            y_min,
                            x_max - x_min,
                            y_max - y_min,
                        ]  # [x, y, w, h]

                        # Create mask dict matching original FAn format
                        mask_dict = {
                            "segmentation": mask,
                            "bbox": bbox,
                            "area": int(mask_area),
                            "point_coords": [x, y],
                            "score": float(score),
                        }

                        # Deduplicate using simple hash
                        mask_hash = hash(mask.tobytes()[:1000])
                        if mask_hash not in seen_masks:
                            all_mask_dicts.append(mask_dict)
                            seen_masks.add(mask_hash)

                            # Limit total masks to avoid memory issues
                            if len(all_mask_dicts) >= 100:
                                break

                if len(all_mask_dicts) >= 100:
                    break

            # Sort by score (original FAn sorts by cfg['sort_by'], default is 'area')
            # We'll sort by score (quality) then by area
            all_mask_dicts.sort(key=lambda x: (x["score"], x["area"]), reverse=True)

            # Return top masks (original FAn doesn't limit, but we limit for efficiency)
            return all_mask_dicts[:50]  # Top 50 masks

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
            # 1. Get masks from SAM 2 (returns list of dicts matching original FAn format)
            mask_dicts = []
            if self.use_sam2 and self.sam2_predictor is not None:
                mask_dicts = self._get_masks_sam2(image_rgb)

            if not mask_dicts:
                # Fallback to simple mask (full image) when SAM 2 not available
                h, w = image_rgb.shape[:2]
                full_mask = np.ones((h, w), dtype=bool)
                mask_dicts = [
                    {
                        "segmentation": full_mask,
                        "bbox": [0, 0, w, h],
                        "area": h * w,
                        "point_coords": [w // 2, h // 2],
                        "score": 1.0,
                    }
                ]

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
            # CRITICAL: For text queries, use CLIP for both text and masked regions (same dimension)
            # DINO is 768-dim, CLIP is 512-dim - they cannot be compared directly!
            # Original FAn: For text queries, use CLIP for masked regions too
            # For image/click queries (not implemented), use DINO for both
            best_mask = None
            best_score = -1.0
            best_mask_features = None  # Store features of best mask for re-detection

            # Log mask count for debugging (use INFO level so it shows up)
            logger.info(
                f"FAn: Evaluating {len(mask_dicts)} masks from SAM 2 for query: '{text_prompt}'"
            )

            if not mask_dicts:
                logger.warning(
                    f"No masks generated by SAM 2 for query: '{text_prompt}'"
                )
                return []

            # Original FAn: Skip first mask (ii == 0) and filter by min_area_size
            # Also extract features from ROI bbox, not full masked image
            MIN_AREA_SIZE = 200  # Original FAn default

            for idx, mask_dict in enumerate(mask_dicts):
                # Skip first mask (original FAn does this: if ii == 0: continue)
                if idx == 0:
                    continue

                # Filter by min_area_size (original FAn: if mask['area'] < min_area_size: continue)
                if mask_dict["area"] < MIN_AREA_SIZE:
                    continue

                # Get mask and bbox from dict (matching original FAn format)
                mask = mask_dict["segmentation"]
                _x, _y, _w, _h = mask_dict["bbox"]  # [x, y, w, h] format

                # Extract ROI from bbox (original FAn approach)
                # Original FAn: img_roi = frameshow[_y : _y + _h, _x : _x + _w, :]
                img_roi = image_rgb[_y : _y + _h, _x : _x + _w, :]

                # For text queries: Use CLIP for ROI (matches CLIP text features - 512-dim)
                # Original FAn: Extract features from ROI bbox, not full masked image
                if self.clip_model is not None:
                    try:
                        # Preprocess ROI image (original FAn approach)
                        pil_image = transforms.ToPILImage()(img_roi)
                        image_tensor = (
                            self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
                        )

                        with torch.no_grad():
                            image_features = self.clip_model.encode_image(image_tensor)
                            image_features = image_features / image_features.norm(
                                dim=-1, keepdim=True
                            )

                        # Compute similarity (both are 512-dim, so this works)
                        similarity = (query_features @ image_features.T).item()

                        if similarity > best_score:
                            best_score = similarity
                            best_mask = mask
                            best_mask_features = (
                                image_features.cpu()
                            )  # Store for re-detection (CLIP features for re-detection)
                    except Exception as e:
                        logger.warning(f"Error encoding image with Open-CLIP: {e}")
                        continue
                # Fallback: If CLIP not available, try DINO (but this won't work with text queries)
                elif self.dino_model is not None:
                    logger.warning(
                        "DINO available but CLIP not available - cannot match text query with DINO features"
                    )
                    continue

            # 4. Convert best mask to bounding box
            # Check if we should use stored features for re-detection
            use_stored_features = (
                len(self._stored_features) > 0 and self._tracking_initialized
            )

            if best_mask is not None:
                logger.info(f"FAn: Found best mask with similarity: {best_score:.3f}")
            else:
                logger.info(
                    f"FAn: No mask matched query '{text_prompt}' (best_score: {best_score:.3f}, threshold: 0.20)"
                )

            if best_mask is not None:
                y_indices, x_indices = np.where(best_mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x_min, x_max = float(x_indices.min()), float(x_indices.max())
                    y_min, y_max = float(y_indices.min()), float(y_indices.max())
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Reject full-frame detections (cover >90% of image - lowered from 95%)
                    # Original FAn uses class_threshold=0.4, but CLIP similarities are lower
                    bbox_area = bbox_width * bbox_height
                    image_area = w * h
                    area_coverage = bbox_area / image_area if image_area > 0 else 0

                    # Check similarity threshold
                    # Original FAn uses class_threshold=0.4 (default) for DINO
                    # For CLIP, similarities are typically lower (0.15-0.30 range)
                    # However, since we extract ROI from bbox (not full masked image),
                    # similarities might be higher. Original FAn uses 0.4 for both.
                    # But based on logs, CLIP similarities are 0.22-0.27, so we use 0.25
                    # as a compromise - still higher than before (0.15) but realistic for CLIP
                    # If using stored features (re-detection), use slightly lower threshold
                    similarity_threshold = 0.22 if use_stored_features else 0.25

                    # Log similarity for debugging (use INFO level so it shows up)
                    logger.info(
                        f"FAn: Best mask similarity: {best_score:.3f} (threshold: {similarity_threshold:.3f}), "
                        f"coverage: {area_coverage:.2%}, masks evaluated: {len(mask_dicts)}"
                    )

                    # Lower coverage threshold from 95% to 90% to allow larger objects
                    # But still reject full-frame detections
                    if best_score > similarity_threshold and area_coverage < 0.90:
                        # If re-detecting with stored features, compare to stored features
                        if use_stored_features and best_mask_features is not None:
                            # Compare to average of stored features
                            avg_stored = torch.stack(self._stored_features).mean(dim=0)
                            stored_similarity = (
                                best_mask_features.to(self.device)
                                @ avg_stored.to(self.device).T
                            ).item()
                            # Use higher of text similarity or stored feature similarity
                            best_score = max(best_score, stored_similarity * 0.8)

                        bbox = BoundingBox(
                            x=x_min,
                            y=y_min,
                            width=bbox_width,
                            height=bbox_height,
                            confidence=float(best_score),
                            class_name=text_prompt,
                        )
                        boxes.append(bbox)

                        # Store tracked object features for re-detection (CLIP features of mask)
                        if best_mask_features is not None:
                            self._stored_features.append(best_mask_features)
                            # Keep only last 30 features to avoid memory issues
                            if len(self._stored_features) > 30:
                                self._stored_features = self._stored_features[-30:]

                        # Store query features for initial detection
                        if self._stored_query_features is None:
                            self._stored_query_features = query_features.cpu()
                    elif area_coverage >= 0.95:
                        logger.debug(
                            f"Rejected full-frame detection (coverage={area_coverage:.2%}, conf={best_score:.2f})"
                        )

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

        Implements automatic re-detection: stores CLIP features of tracked object
        at every frame, uses them for re-detection when tracking is lost.
        Uses CLIP features (not DINO) to match text query features (both 512-dim).

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
            state = self.tracker.update(frame, initial_bbox)

            # Store tracked object features for automatic re-detection
            # Use CLIP features (same as detection) to ensure dimension compatibility
            if state is not None and self.clip_model is not None:
                try:
                    # Extract CLIP features from tracked region (matches detection features)
                    x, y, w, h = state.bbox
                    # Convert to int and clamp to image bounds
                    h_img, w_img = frame.shape[:2]
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(w_img, int(x + w))
                    y2 = min(h_img, int(y + h))

                    if x2 > x1 and y2 > y1:
                        # Extract region
                        tracked_region = frame[y1:y2, x1:x2]
                        if tracked_region.size > 0:
                            # Convert BGR to RGB
                            import cv2

                            tracked_region_rgb = cv2.cvtColor(
                                tracked_region, cv2.COLOR_BGR2RGB
                            )
                            pil_image = transforms.ToPILImage()(
                                tracked_region_rgb.astype(np.uint8)
                            )

                            # Extract CLIP features (same as detection, 512-dim)
                            image_tensor = (
                                self.clip_preprocess(pil_image)
                                .unsqueeze(0)
                                .to(self.device)
                            )

                            with torch.no_grad():
                                track_features = self.clip_model.encode_image(
                                    image_tensor
                                )
                                track_features = track_features / track_features.norm(
                                    dim=-1, keepdim=True
                                )

                            # Store features for re-detection (CLIP features, 512-dim)
                            self._stored_features.append(track_features.cpu())
                            # Keep only last 30 features
                            if len(self._stored_features) > 30:
                                self._stored_features = self._stored_features[-30:]

                            self._last_tracked_bbox = state.bbox
                            self._tracking_initialized = True
                except Exception as e:
                    logger.debug(f"Error storing tracked object features: {e}")

            # Initialize tracking on first detection
            if initial_bbox is not None:
                self._tracking_initialized = True

            return state

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
        self._stored_query_features = None
        self._current_image = None
        self._last_tracked_bbox = None
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
            SmartTracker,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = GroundingDINOModel(device=self.device)
        base_tracker = BotSORTTracker(device=self.device)
        # Wrap with SmartTracker for automatic re-detection logic
        self.tracker = SmartTracker(
            base_tracker=base_tracker,
            detection_callback=None,  # Manual re-detection via update() with initial_bbox
            min_hits_to_confirm=5,
            redetect_interval=10,
        )
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
