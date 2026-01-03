#!/usr/bin/env python3
"""Test Follow Anything (FAn) components individually."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
import numpy as np
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import (
    FollowAnythingModel,
)


def test_open_clip():
    """Test Open-CLIP loading."""
    logger.info("=" * 60)
    logger.info("Testing Open-CLIP")
    logger.info("=" * 60)

    try:
        import open_clip  # type: ignore

        logger.info("✓ open-clip-torch imported successfully")

        # Try to load a model
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        logger.info("✓ Open-CLIP model loaded: ViT-B-32")

        # Test tokenizer
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info("✓ Open-CLIP tokenizer loaded")

        return True
    except ImportError as e:
        logger.error(f"✗ open-clip-torch not available: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error loading Open-CLIP: {e}")
        return False


def test_sam2():
    """Test SAM 2 loading."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing SAM 2")
    logger.info("=" * 60)

    try:
        import sam2  # type: ignore

        logger.info("✓ sam2 imported successfully")
        logger.info(f"  SAM 2 location: {sam2.__file__}")

        # Check if we can import the build function
        try:
            from sam2.build_sam import build_sam2  # type: ignore  # noqa: F401
            from sam2.sam2_image_predictor import (
                SAM2ImagePredictor,  # type: ignore  # noqa: F401
            )

            logger.info("✓ SAM 2 modules imported successfully")
            logger.info(
                "  Note: Full initialization requires checkpoint (set SAM2_CHECKPOINT env var)"
            )
            return True
        except ImportError as e:
            logger.warning(f"✗ SAM 2 modules not available: {e}")
            return False
    except ImportError as e:
        logger.error(f"✗ sam2 not available: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error loading SAM 2: {e}")
        return False


def test_fan_model(device: str = "cpu"):
    """Test Follow Anything model initialization."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Follow Anything Model")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    try:
        fan = FollowAnythingModel(device=device)
        logger.info("✓ FollowAnythingModel initialized")

        logger.info(f"  Available: {fan.is_available()}")
        logger.info(f"  Has CLIP: {fan.clip_model is not None}")
        logger.info(f"  Has DINO: {fan.dino_model is not None}")
        logger.info(f"  Has SAM2: {fan.sam2_predictor is not None}")
        logger.info(f"  Has Tracker: {fan.tracker is not None}")
        logger.info(f"  Using fallback: {fan.fallback is not None}")

        # Test detection with fallback
        if fan.fallback is not None:
            test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            boxes = fan.detect(test_image, "test object")
            logger.info(f"  Detection test: {len(boxes)} boxes found")

        return True
    except Exception as e:
        logger.error(f"✗ Error testing FAn model: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all component tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Follow Anything components")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    results = {}

    results["open_clip"] = test_open_clip()
    results["sam2"] = test_sam2()
    results["fan_model"] = test_fan_model(device=args.device)

    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for component, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{component}: {status}")

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
