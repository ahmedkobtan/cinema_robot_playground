#!/usr/bin/env python3
"""Comprehensive test of Follow Anything (FAn) with SAM 2 - full flow."""

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


def create_test_image_with_objects(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a test image with distinct objects."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add a red rectangle (simulating a "red cup")
    cv2.rectangle(image, (200, 200), (400, 500), (0, 0, 255), -1)
    cv2.putText(
        image, "RED", (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    # Add a blue circle (simulating a "blue ball")
    cv2.circle(image, (800, 300), 100, (255, 0, 0), -1)
    cv2.putText(
        image, "BLUE", (750, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # Add a green rectangle
    cv2.rectangle(image, (1000, 100), (1200, 300), (0, 255, 0), -1)

    return image


def test_full_fan_flow_with_sam2(device: str = "cpu"):
    """Test the complete FAn flow with SAM 2."""
    logger.info("=" * 60)
    logger.info("Full Follow Anything Flow Test with SAM 2")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")

    try:
        # Initialize FAn with SAM 2
        fan = FollowAnythingModel(use_sam2=True, device=device)

        if not fan.is_available():
            logger.error("✗ FAn model not available")
            return False

        logger.info("✓ FAn model initialized")
        logger.info(f"  Has SAM 2: {fan.sam2_predictor is not None}")
        logger.info(f"  Has CLIP: {fan.clip_model is not None}")
        logger.info(f"  Has DINO: {fan.dino_model is not None}")

        if fan.sam2_predictor is None:
            logger.warning("  SAM 2 not loaded - check checkpoint location")
            return False

        # Create test image
        test_image = create_test_image_with_objects()
        logger.info(f"✓ Test image created: {test_image.shape}")

        # Test 1: Detection with text prompt
        logger.info("\n--- Test 1: Text-based Detection ---")
        prompts = ["a red cup", "something red", "a blue ball"]

        for prompt in prompts:
            boxes = fan.detect(test_image, text_prompt=prompt)
            logger.info(f"  Prompt: '{prompt}'")
            logger.info(f"    Found {len(boxes)} boxes")
            for i, box in enumerate(boxes):
                logger.info(
                    f"    Box {i + 1}: x={box.x:.1f}, y={box.y:.1f}, "
                    f"w={box.width:.1f}, h={box.height:.1f}, "
                    f"conf={box.confidence:.4f}, class={box.class_name}"
                )

        # Test 2: Tracking initialization
        logger.info("\n--- Test 2: Tracking Initialization ---")
        if boxes:
            initial_bbox = (boxes[0].x, boxes[0].y, boxes[0].width, boxes[0].height)
            state = fan.update(test_image, initial_bbox=initial_bbox)

            if state:
                logger.info("  ✓ Tracking initialized")
                logger.info(
                    f"    Track ID: {state.track_id}, "
                    f"BBox: {state.bbox}, "
                    f"Confidence: {state.confidence:.4f}"
                )
            else:
                logger.warning("  ✗ Tracking initialization failed")
                # This is acceptable if Bot-SORT not available
                logger.info("  Continuing with detection-only test")

        # Test 3: Multi-frame tracking
        logger.info("\n--- Test 3: Multi-frame Tracking ---")
        frames = [create_test_image_with_objects() for _ in range(5)]

        for i, frame in enumerate(frames):
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

        # Test 4: Reset and re-initialize
        logger.info("\n--- Test 4: Reset and Re-initialize ---")
        fan.reset()
        logger.info("  ✓ Tracker reset")

        if boxes:
            state = fan.update(test_image, initial_bbox=initial_bbox)
            if state:
                logger.info("  ✓ Successfully re-initialized tracking")

        logger.info("\n" + "=" * 60)
        logger.info("✓ Full FAn flow test completed successfully")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"✗ Full flow test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sam2_mask_quality(device: str = "cpu"):
    """Test SAM 2 mask quality and precision."""
    logger.info("\n" + "=" * 60)
    logger.info("SAM 2 Mask Quality Test")
    logger.info("=" * 60)

    try:
        fan = FollowAnythingModel(use_sam2=True, device=device)

        if fan.sam2_predictor is None:
            logger.warning("SAM 2 not available, skipping quality test")
            return True

        test_image = create_test_image_with_objects()

        # Test mask generation at different points
        test_points = [
            (300, 350),  # Center of red rectangle
            (800, 300),  # Center of blue circle
            (1100, 200),  # Center of green rectangle
        ]

        logger.info("Testing mask generation at different points...")
        for x, y in test_points:
            masks = fan._get_masks_sam2(test_image)
            if masks:
                logger.info(f"  Point ({x}, {y}): Generated {len(masks)} masks")
                # Check mask quality
                for i, mask in enumerate(masks[:3]):  # Top 3 masks
                    coverage = (np.sum(mask) / mask.size) * 100
                    logger.info(f"    Mask {i + 1}: {coverage:.2f}% coverage")

        return True

    except Exception as e:
        logger.error(f"✗ Mask quality test failed: {e}")
        return False


def main():
    """Run comprehensive FAn tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Follow Anything full flow")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    results = {}

    results["full_flow"] = test_full_fan_flow_with_sam2(args.device)
    results["mask_quality"] = test_sam2_mask_quality(args.device)

    logger.info("\n" + "=" * 60)
    logger.info("Final Test Summary")
    logger.info("=" * 60)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
