#!/usr/bin/env python3
"""Benchmark models on RTX 2080 Ti for Phase 1."""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
import numpy as np
import torch
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    GroundingDINOModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import (
    FollowAnythingModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import (
    BotSORTTracker,
)


class ModelBenchmark:
    """Benchmark models for performance on RTX 2080 Ti."""

    def __init__(
        self, num_frames: int = 100, frame_size: tuple = (720, 1280), device: str = None
    ):
        """
        Initialize benchmark.

        Args:
            num_frames: Number of frames to process
            frame_size: Frame size (height, width)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate_test_frame(self) -> np.ndarray:
        """Generate a test frame."""
        h, w = self.frame_size
        # Generate random image
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        return frame

    def benchmark_follow_anything(self) -> dict:
        """Benchmark Follow Anything model (with SAM 2 if available, else fallback)."""
        logger.info("Benchmarking Follow Anything (FAn)...")

        try:
            # Set SAM2_CHECKPOINT if checkpoint exists in resources
            import os

            if not os.getenv("SAM2_CHECKPOINT"):
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
                        os.environ["SAM2_CHECKPOINT"] = str(checkpoint_path)
                        logger.info(f"Using SAM 2 checkpoint: {checkpoint_path.name}")
                        break

            # Try to use SAM 2 if checkpoint is available, otherwise use fallback
            model = FollowAnythingModel(use_sam2=True, device=self.device)
            if not model.is_available() and model.fallback is None:
                logger.error("Follow Anything not available")
                return {"available": False, "error": "Model not available"}

            # Warmup
            test_frame = self.generate_test_frame()
            _ = model.detect(test_frame, "test object")
            time.sleep(1)

            # Benchmark detection
            detection_times = []
            for _ in range(self.num_frames):
                frame = self.generate_test_frame()
                start = time.time()
                _ = model.detect(frame, "test object")
                detection_times.append(time.time() - start)

            avg_detection_time = np.mean(detection_times)
            fps_detection = 1.0 / avg_detection_time if avg_detection_time > 0 else 0

            # Benchmark tracking
            tracking_times = []
            initial_bbox = (100.0, 100.0, 200.0, 200.0)
            model.update(test_frame, initial_bbox)

            for _ in range(self.num_frames):
                frame = self.generate_test_frame()
                start = time.time()
                _ = model.update(frame)
                tracking_times.append(time.time() - start)

            avg_tracking_time = np.mean(tracking_times)
            fps_tracking = 1.0 / avg_tracking_time if avg_tracking_time > 0 else 0

            # VRAM usage
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                vram_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            else:
                vram_allocated = 0
                vram_reserved = 0

            return {
                "available": True,
                "detection_fps": fps_detection,
                "detection_latency_ms": avg_detection_time * 1000,
                "tracking_fps": fps_tracking,
                "tracking_latency_ms": avg_tracking_time * 1000,
                "vram_allocated_gb": vram_allocated,
                "vram_reserved_gb": vram_reserved,
            }

        except Exception as e:
            logger.error(f"Error benchmarking Follow Anything: {e}")
            return {"available": False, "error": str(e)}

    def benchmark_grounding_dino(self) -> dict:
        """Benchmark Grounding DINO model."""
        logger.info("Benchmarking Grounding DINO...")

        try:
            model = GroundingDINOModel(device=self.device)

            # Try to load model by attempting detection
            test_frame = self.generate_test_frame()
            _test_boxes = model.detect(test_frame, "test object")

            # Check if model loaded successfully
            if not model.is_available():
                logger.warning("Grounding DINO not available after loading attempt")
                return {"available": False, "error": "Model not available"}

            # Warmup
            _ = model.detect(test_frame, "test object")
            time.sleep(1)

            # Benchmark
            detection_times = []
            for _ in range(self.num_frames):
                frame = self.generate_test_frame()
                start = time.time()
                _ = model.detect(frame, "test object")
                detection_times.append(time.time() - start)

            avg_time = np.mean(detection_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0

            # VRAM usage
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                vram_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            else:
                vram_allocated = 0
                vram_reserved = 0

            return {
                "available": True,
                "fps": fps,
                "latency_ms": avg_time * 1000,
                "vram_allocated_gb": vram_allocated,
                "vram_reserved_gb": vram_reserved,
            }

        except Exception as e:
            logger.error(f"Error benchmarking Grounding DINO: {e}")
            return {"available": False, "error": str(e)}

    def benchmark_botsort(self) -> dict:
        """Benchmark Bot-SORT tracker."""
        logger.info("Benchmarking Bot-SORT...")

        try:
            tracker = BotSORTTracker(device=self.device)
            if not tracker.is_available():
                logger.warning("Bot-SORT not available")
                return {"available": False, "error": "Tracker not available"}

            # Initialize
            test_frame = self.generate_test_frame()
            initial_bbox = (100.0, 100.0, 200.0, 200.0)
            tracker.update(test_frame, initial_bbox)
            time.sleep(1)

            # Benchmark
            tracking_times = []
            for _ in range(self.num_frames):
                frame = self.generate_test_frame()
                start = time.time()
                _ = tracker.update(frame)
                tracking_times.append(time.time() - start)

            avg_time = np.mean(tracking_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0

            # VRAM usage
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                vram_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            else:
                vram_allocated = 0
                vram_reserved = 0

            return {
                "available": True,
                "fps": fps,
                "latency_ms": avg_time * 1000,
                "vram_allocated_gb": vram_allocated,
                "vram_reserved_gb": vram_reserved,
            }

        except Exception as e:
            logger.error(f"Error benchmarking Bot-SORT: {e}")
            return {"available": False, "error": str(e)}

    def run_all_benchmarks(self) -> dict:
        """Run all benchmarks and return results."""
        logger.info("=" * 60)
        logger.info("Model Benchmarking on RTX 2080 Ti")
        logger.info("=" * 60)

        results = {}

        # Check GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(
                f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
            )
        else:
            logger.warning("CUDA not available - benchmarks will run on CPU")

        # Benchmark Follow Anything
        results["follow_anything"] = self.benchmark_follow_anything()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Benchmark Grounding DINO
        results["grounding_dino"] = self.benchmark_grounding_dino()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Benchmark Bot-SORT
        results["botsort"] = self.benchmark_botsort()

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Benchmark Results Summary")
        logger.info("=" * 60)

        for model_name, result in results.items():
            logger.info(f"\n{model_name.upper()}:")
            if result.get("available"):
                if "detection_fps" in result:
                    logger.info(f"  Detection FPS: {result['detection_fps']:.2f}")
                    logger.info(
                        f"  Detection Latency: {result['detection_latency_ms']:.2f} ms"
                    )
                    logger.info(f"  Tracking FPS: {result['tracking_fps']:.2f}")
                    logger.info(
                        f"  Tracking Latency: {result['tracking_latency_ms']:.2f} ms"
                    )
                else:
                    logger.info(f"  FPS: {result['fps']:.2f}")
                    logger.info(f"  Latency: {result['latency_ms']:.2f} ms")
                logger.info(f"  VRAM Allocated: {result['vram_allocated_gb']:.2f} GB")
                logger.info(f"  VRAM Reserved: {result['vram_reserved_gb']:.2f} GB")
            else:
                logger.warning(
                    f"  Not available: {result.get('error', 'Unknown error')}"
                )

        return results


def main():
    """Run benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark models for Cinema Bot")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default="720x1280",
        help="Frame size as HxW (default: 720x1280)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu', default: auto-detect)",
    )

    args = parser.parse_args()

    # Parse frame size
    h, w = map(int, args.frame_size.split("x"))
    frame_size = (h, w)

    benchmark = ModelBenchmark(
        num_frames=args.num_frames, frame_size=frame_size, device=args.device
    )
    results = benchmark.run_all_benchmarks()

    # Save results to file
    import json

    results_file = Path(__file__).parent.parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    return 0 if all(r.get("available", False) for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
