#!/usr/bin/env python3
"""Test SAM 2 checkpoint loading and functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
import cv2
import numpy as np
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import (
    FollowAnythingModel,
)


def create_test_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a test image with some objects."""
    # Create a colorful test image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add a red rectangle (simulating a "red cup")
    cv2.rectangle(image, (200, 200), (400, 500), (0, 0, 255), -1)

    # Add a blue circle (simulating a "blue ball")
    cv2.circle(image, (800, 300), 100, (255, 0, 0), -1)

    # Add some text
    cv2.putText(
        image, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
    )

    return image


def test_sam2_checkpoint_loading(checkpoint_path: str, device: str = "cpu"):
    """Test SAM 2 checkpoint loading."""
    logger.info("=" * 60)
    logger.info("Testing SAM 2 Checkpoint Loading")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Device: {device}")

    import os

    if not os.path.exists(checkpoint_path):
        logger.error(f"✗ Checkpoint file not found: {checkpoint_path}")
        return False

    logger.info(
        f"✓ Checkpoint file exists ({os.path.getsize(checkpoint_path) / (1024**2):.1f} MB)"
    )

    try:
        import sam2  # type: ignore
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        logger.info("✓ SAM 2 package imported")

        # Try to find config file
        sam2_config_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
        logger.info(f"Config directory: {sam2_config_dir}")

        # Auto-detect config based on checkpoint filename
        checkpoint_name = os.path.basename(checkpoint_path).lower()
        logger.info(f"Checkpoint name: {checkpoint_name}")

        config_mapping = {
            "tiny": "configs/sam2.1/sam2.1_hiera_t",
            "small": "configs/sam2.1/sam2.1_hiera_s",
            "base_plus": "configs/sam2.1/sam2.1_hiera_b+",
            "base": "configs/sam2.1/sam2.1_hiera_b+",
            "large": "configs/sam2.1/sam2.1_hiera_l",
        }

        sam2_model_cfg = None
        for key, config_name in config_mapping.items():
            if key in checkpoint_name:
                # Verify config file exists (remove configs/ prefix for file check)
                cfg_file_name = config_name.replace("configs/", "")
                cfg_path = os.path.join(sam2_config_dir, cfg_file_name + ".yaml")
                if os.path.exists(cfg_path):
                    sam2_model_cfg = config_name  # Use full path with configs/ prefix
                    logger.info(f"✓ Found config: {config_name}")
                    break

        if sam2_model_cfg is None:
            logger.warning("Config file not found, trying fallback...")
            for cfg_name in [
                "configs/sam2.1/sam2.1_hiera_b+",
                "configs/sam2.1/sam2.1_hiera_l",
                "configs/sam2.1/sam2.1_hiera_s",
                "configs/sam2.1/sam2.1_hiera_t",
            ]:
                cfg_file_name = cfg_name.replace("configs/", "")
                cfg_path = os.path.join(sam2_config_dir, cfg_file_name + ".yaml")
                if os.path.exists(cfg_path):
                    sam2_model_cfg = cfg_name
                    logger.info(f"✓ Using fallback config: {cfg_name}")
                    break

        if sam2_model_cfg is None:
            logger.error("✗ Could not find config file")
            logger.info("Available configs:")
            if os.path.exists(sam2_config_dir):
                for item in os.listdir(sam2_config_dir):
                    logger.info(f"  {item}")
            return False

        # Build model
        logger.info("Building SAM 2 model...")
        sam2_model = build_sam2(sam2_model_cfg, checkpoint_path, device=device)
        logger.info("✓ Model built successfully")

        # Create predictor
        _predictor = SAM2ImagePredictor(sam2_model)
        logger.info("✓ Predictor created successfully")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to load SAM 2: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sam2_mask_generation(checkpoint_path: str, device: str = "cpu"):
    """Test SAM 2 mask generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing SAM 2 Mask Generation")
    logger.info("=" * 60)

    try:
        import os

        import sam2  # type: ignore
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        # Find config
        sam2_config_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
        checkpoint_name = os.path.basename(checkpoint_path).lower()

        config_mapping = {
            "tiny": "configs/sam2.1/sam2.1_hiera_t",
            "small": "configs/sam2.1/sam2.1_hiera_s",
            "base_plus": "configs/sam2.1/sam2.1_hiera_b+",
            "base": "configs/sam2.1/sam2.1_hiera_b+",
            "large": "configs/sam2.1/sam2.1_hiera_l",
        }

        sam2_model_cfg = None
        for key, config_name in config_mapping.items():
            if key in checkpoint_name:
                cfg_file_name = config_name.replace("configs/", "")
                cfg_path = os.path.join(sam2_config_dir, cfg_file_name + ".yaml")
                if os.path.exists(cfg_path):
                    sam2_model_cfg = config_name
                    break

        if sam2_model_cfg is None:
            for cfg_name in [
                "configs/sam2.1/sam2.1_hiera_b+",
                "configs/sam2.1/sam2.1_hiera_l",
            ]:
                cfg_file_name = cfg_name.replace("configs/", "")
                cfg_path = os.path.join(sam2_config_dir, cfg_file_name + ".yaml")
                if os.path.exists(cfg_path):
                    sam2_model_cfg = cfg_name
                    break

        # Build model and predictor
        sam2_model = build_sam2(sam2_model_cfg, checkpoint_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        # Create test image
        test_image = create_test_image()
        logger.info(f"Test image shape: {test_image.shape}")

        # Set image
        predictor.set_image(test_image)
        logger.info("✓ Image set in predictor")

        # Generate masks from points
        points = np.array(
            [[640, 360], [300, 350]], dtype=np.float32
        )  # Center and red rectangle
        point_labels = np.array([1, 1], dtype=np.int32)

        logger.info(f"Generating masks from {len(points)} points...")
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            multimask_output=True,
        )

        logger.info(f"✓ Generated {len(masks)} masks")
        logger.info(f"  Scores: {scores}")
        logger.info(f"  Mask shapes: {[m.shape for m in masks]}")

        # Check mask validity
        for i, mask in enumerate(masks):
            mask_area = np.sum(mask)
            total_area = mask.shape[0] * mask.shape[1]
            coverage = (mask_area / total_area) * 100
            logger.info(
                f"  Mask {i + 1}: {coverage:.2f}% coverage, score={scores[i]:.4f}"
            )

        return True

    except Exception as e:
        logger.error(f"✗ Failed to generate masks: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fan_with_sam2(checkpoint_path: str, device: str = "cpu"):
    """Test Follow Anything with SAM 2."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Follow Anything with SAM 2")
    logger.info("=" * 60)

    import os

    os.environ["SAM2_CHECKPOINT"] = checkpoint_path

    try:
        fan = FollowAnythingModel(use_sam2=True, device=device)

        if fan.sam2_predictor is None:
            logger.error("✗ SAM 2 predictor not loaded")
            return False

        logger.info("✓ Follow Anything with SAM 2 loaded")

        # Test detection
        test_image = create_test_image()
        boxes = fan.detect(test_image, text_prompt="a red cup")

        logger.info(f"✓ Detection test: {len(boxes)} boxes found")
        for i, box in enumerate(boxes):
            logger.info(
                f"  Box {i + 1}: x={box.x:.1f}, y={box.y:.1f}, "
                f"w={box.width:.1f}, h={box.height:.1f}, "
                f"conf={box.confidence:.4f}"
            )

        return True

    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run SAM 2 checkpoint tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test SAM 2 checkpoint")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to SAM 2 checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SAM 2 Checkpoint Testing")
    logger.info("=" * 60)

    results = {}

    results["loading"] = test_sam2_checkpoint_loading(args.checkpoint_path, args.device)
    if results["loading"]:
        results["mask_generation"] = test_sam2_mask_generation(
            args.checkpoint_path, args.device
        )
        results["fan_integration"] = test_fan_with_sam2(
            args.checkpoint_path, args.device
        )

    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
