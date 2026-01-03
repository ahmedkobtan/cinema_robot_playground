#!/usr/bin/env python3
"""Test script to verify Cinema Bot setup."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
import numpy as np

from ahmedkobtan_cinema_robot_playground.src.algorithms.pid_controller import (
    PIDController,
)
from ahmedkobtan_cinema_robot_playground.src.algorithms.shot_planner import (
    ShotPlanner,
    ShotType,
)
from ahmedkobtan_cinema_robot_playground.src.algorithms.visual_servoing import (
    VisualServoingController,
)
from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    MockDetectionModel,
)
from ahmedkobtan_cinema_robot_playground.src.models.tracking_models import MockTracker
from ahmedkobtan_cinema_robot_playground.src.services.cinematographer_agent import (
    CinematographerAgent,
)
from ahmedkobtan_cinema_robot_playground.src.services.director_agent import (
    DirectorAgent,
)


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_pid_controller():
    """Test PID controller."""
    print("\nTesting PID controller...")
    try:
        pid = PIDController(kp=1.0, ki=0.1, kd=0.01)
        output = pid.compute(10.0)
        assert output > 0
        print("✓ PID controller works")
        return True
    except Exception as e:
        print(f"✗ PID controller error: {e}")
        return False


def test_visual_servoing():
    """Test visual servoing."""
    print("\nTesting visual servoing...")
    try:
        controller = VisualServoingController(frame_width=1280, frame_height=720)
        servo_pulse, wheel_left, wheel_right, wheel_forward = (
            controller.compute_commands(700.0, 360.0, 100.0, 100.0)
        )
        assert 900 <= servo_pulse <= 2100
        print("✓ Visual servoing works")
        return True
    except Exception as e:
        print(f"✗ Visual servoing error: {e}")
        return False


def test_shot_planner():
    """Test shot planner."""
    print("\nTesting shot planner...")
    try:
        planner = ShotPlanner(frame_width=1280, frame_height=720)
        trajectory = planner.plan_shot(ShotType.FOLLOW, 640.0, 360.0, 100.0, 100.0)
        assert "shot_type" in trajectory
        print("✓ Shot planner works")
        return True
    except Exception as e:
        print(f"✗ Shot planner error: {e}")
        return False


def test_director_agent(device: str = "cpu"):
    """Test Director Agent."""
    print("\nTesting Director Agent...")
    try:
        agent = DirectorAgent(
            detection_model=MockDetectionModel(), use_llm=False, device=device
        )
        result = agent.parse_command("Orbit the red cup")
        assert "object" in result
        assert "shot_type" in result
        print("✓ Director Agent works")
        return True
    except Exception as e:
        print(f"✗ Director Agent error: {e}")
        return False


def test_cinematographer_agent(device: str = "cpu"):
    """Test Cinematographer Agent."""
    print("\nTesting Cinematographer Agent...")
    try:
        agent = CinematographerAgent(tracker=MockTracker(), device=device)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        initial_bbox = (100.0, 100.0, 200.0, 200.0)
        success = agent.initialize_tracking(frame, initial_bbox)
        assert success
        print("✓ Cinematographer Agent works")
        return True
    except Exception as e:
        print(f"✗ Cinematographer Agent error: {e}")
        return False


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Cinema Bot setup")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Cinema Bot Setup Test")
    print(f"Device: {args.device}")
    print("=" * 50)

    tests = [
        test_imports,
        test_pid_controller,
        test_visual_servoing,
        test_shot_planner,
        lambda: test_director_agent(device=args.device),
        lambda: test_cinematographer_agent(device=args.device),
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 50)

    if all(results):
        print("\n✓ All tests passed! Setup is correct.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
