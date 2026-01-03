#!/usr/bin/env python3
"""End-to-end test for Follow Anything (FAn) with actual functionality testing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
import cv2
import numpy as np
import torch
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


def test_clip_text_encoding(device: str = "cpu"):
    """Test Open-CLIP text encoding."""
    logger.info("=" * 60)
    logger.info("Test 1: Open-CLIP Text Encoding")
    logger.info("=" * 60)

    try:
        import open_clip  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Test encoding text
        text = "a red cup"
        text_tokens = tokenizer([text]).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logger.info(f"✓ Text encoded: '{text}'")
        logger.info(f"  Feature shape: {text_features.shape}")
        logger.info(f"  Feature norm: {text_features.norm().item():.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_clip_image_encoding(device: str = "cpu"):
    """Test Open-CLIP image encoding."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Open-CLIP Image Encoding")
    logger.info("=" * 60)

    try:
        import open_clip  # type: ignore
        import torchvision.transforms as transforms

        model, preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )

        # Create test image
        test_image = create_test_image()
        pil_image = transforms.ToPILImage()(test_image)

        # Preprocess and encode
        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logger.info("✓ Image encoded")
        logger.info(f"  Feature shape: {image_features.shape}")
        logger.info(f"  Feature norm: {image_features.norm().item():.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_clip_similarity(device: str = "cpu"):
    """Test Open-CLIP text-image similarity."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Open-CLIP Text-Image Similarity")
    logger.info("=" * 60)

    try:
        import open_clip  # type: ignore
        import torch
        import torchvision.transforms as transforms

        model, preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Create test image
        test_image = create_test_image()
        pil_image = transforms.ToPILImage()(test_image)
        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        # Test multiple text prompts
        prompts = ["a red cup", "a blue ball", "a green car", "a test image"]

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for prompt in prompts:
                text_tokens = tokenizer([prompt]).to(device)
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (text_features @ image_features.T).item()
                logger.info(f"  '{prompt}': similarity = {similarity:.4f}")

        logger.info("✓ Text-image similarity computed")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fan_detection_without_sam(device: str = "cpu"):
    """Test FAn detection without SAM (using CLIP + full image)."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: FAn Detection (without SAM)")
    logger.info("=" * 60)

    try:
        fan = FollowAnythingModel(use_sam2=False, device=device)

        if not fan.is_available():
            logger.warning("FAn model not available, skipping test")
            return False

        # Create test image
        test_image = create_test_image()

        # Test detection with different prompts
        prompts = ["a red cup", "a blue ball", "something red"]

        for prompt in prompts:
            boxes = fan.detect(test_image, text_prompt=prompt)
            logger.info(f"  Prompt: '{prompt}' -> {len(boxes)} boxes found")
            for i, box in enumerate(boxes):
                logger.info(
                    f"    Box {i + 1}: x={box.x:.1f}, y={box.y:.1f}, "
                    f"w={box.width:.1f}, h={box.height:.1f}, "
                    f"conf={box.confidence:.4f}"
                )

        logger.info("✓ FAn detection (without SAM) works")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fan_tracking(device: str = "cpu"):
    """Test FAn tracking."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: FAn Tracking")
    logger.info("=" * 60)

    try:
        fan = FollowAnythingModel(device=device)

        # Create test frames
        frames = [create_test_image() for _ in range(5)]

        # Initialize tracking
        initial_bbox = (200.0, 200.0, 200.0, 300.0)  # x, y, width, height
        state = fan.update(frames[0], initial_bbox=initial_bbox)

        if state:
            logger.info(
                f"  Initial state: track_id={state.track_id}, "
                f"bbox={state.bbox}, confidence={state.confidence:.4f}"
            )
        else:
            logger.warning("  Initial tracking state is None")
            return False

        # Continue tracking
        for i, frame in enumerate(frames[1:], 1):
            state = fan.update(frame)
            if state:
                logger.info(
                    f"  Frame {i + 1}: track_id={state.track_id}, "
                    f"bbox={state.bbox}, confidence={state.confidence:.4f}"
                )
            else:
                logger.warning(f"  Frame {i + 1}: tracking lost")
                # This is acceptable - tracking can be lost
                break

        logger.info("✓ FAn tracking works")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sam2_if_available(device: str = "cpu"):
    """Test SAM 2 if checkpoint is available."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: SAM 2 (if checkpoint available)")
    logger.info("=" * 60)

    import os
    from pathlib import Path

    # Check multiple locations for checkpoint
    sam2_checkpoint = os.getenv("SAM2_CHECKPOINT", None)

    # Check resources directory (default location)
    if not sam2_checkpoint or not os.path.exists(sam2_checkpoint):
        project_root = Path(__file__).parent.parent
        resources_dir = (
            project_root / "ahmedkobtan_cinema_robot_playground" / "resources"
        )
        default_checkpoints = [
            resources_dir / "sam2.1_hiera_base_plus.pt",
            resources_dir / "sam2.1_hiera_large.pt",
            resources_dir / "sam2.1_hiera_small.pt",
            resources_dir / "sam2.1_hiera_tiny.pt",
        ]

        for checkpoint_path in default_checkpoints:
            if checkpoint_path.exists():
                sam2_checkpoint = str(checkpoint_path)
                logger.info(f"  Found checkpoint in resources: {checkpoint_path.name}")
                break

    if not sam2_checkpoint or not os.path.exists(sam2_checkpoint):
        logger.info("  SAM2_CHECKPOINT not set or file not found, skipping")
        logger.info(
            "  To test SAM 2, set: export SAM2_CHECKPOINT=/path/to/checkpoint.pth"
        )
        logger.info(
            "  Or place checkpoint in: ahmedkobtan_cinema_robot_playground/resources/"
        )
        return True  # Not a failure, just not available

    try:
        import os

        import sam2  # type: ignore
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        # Find config file based on checkpoint name
        sam2_config_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
        checkpoint_name = os.path.basename(sam2_checkpoint).lower()

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
                    logger.info(f"  Using config: {config_name}")
                    break

        if sam2_model_cfg is None:
            # Fallback to base_plus
            sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_b+"
            logger.info(f"  Using fallback config: {sam2_model_cfg}")

        # Build model
        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        # Test on image
        test_image = create_test_image()
        predictor.set_image(test_image)

        # Generate masks
        points = np.array([[640, 360]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            multimask_output=True,
        )

        logger.info(f"✓ SAM 2 generated {len(masks)} masks")
        logger.info(f"  Scores: {scores}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all end-to-end tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Follow Anything end-to-end")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    device = args.device
    logger.info("=" * 60)
    logger.info("Follow Anything (FAn) End-to-End Tests")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    results = {}

    results["clip_text"] = test_clip_text_encoding(device)
    results["clip_image"] = test_clip_image_encoding(device)
    results["clip_similarity"] = test_clip_similarity(device)
    results["fan_detection"] = test_fan_detection_without_sam(device)
    results["fan_tracking"] = test_fan_tracking(device)
    results["sam2"] = test_sam2_if_available(device)

    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    import cv2
    import torch

    sys.exit(main())
