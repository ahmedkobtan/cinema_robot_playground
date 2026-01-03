#!/usr/bin/env python3
"""Test Transformers-based LLM setup for Director Agent.

NOTE: This script is kept for backward compatibility but now uses Transformers.
For comprehensive testing, use scripts/test_llm_transformers.py instead.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.services.director_agent import (
    DirectorAgent,
)


def test_llm_agent(device: str = "cpu"):
    """Test LLM agent setup and command parsing."""
    logger.info("=" * 60)
    logger.info("Testing Transformers-based LLM Setup")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # Test without LLM (simple parsing)
    logger.info("\n1. Testing simple command parsing (no LLM)...")
    agent_simple = DirectorAgent(use_llm=False, device=device)
    test_commands = [
        "Orbit the red cup",
        "Follow the cat",
        "Dolly in on the object",
        "Track the person in the green shirt",
    ]

    for cmd in test_commands:
        result = agent_simple.parse_command(cmd)
        logger.info(f"  Command: '{cmd}'")
        logger.info(f"  Parsed: {result}")

    # Test with LLM (Transformers - if available)
    logger.info("\n2. Testing Transformers-based LLM command parsing...")
    try:
        logger.info("  Loading LLM model (this may take a few minutes)...")
        agent_llm = DirectorAgent(use_llm=True, device=device)
        if agent_llm.llm_agent is not None:
            logger.info("  ✓ LLM agent initialized successfully")

            for cmd in test_commands:
                result = agent_llm.parse_command(cmd)
                logger.info(f"  Command: '{cmd}'")
                logger.info(f"  Parsed: {result}")
        else:
            logger.warning("  ✗ LLM agent not available")
            logger.info(
                "  This is expected if model download fails or CUDA unavailable"
            )
            logger.info("  System will fall back to simple parsing")

    except Exception as e:
        logger.error(f"  ✗ Error setting up LLM agent: {e}")
        logger.info("  Falling back to simple parsing (this is acceptable)")

    logger.info("\n" + "=" * 60)
    logger.info("LLM Agent Test Complete")
    logger.info("=" * 60)


def main():
    """Run LLM agent tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM agent for Director")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    test_llm_agent(device=args.device)
    logger.info(
        "\nNOTE: For comprehensive LLM testing, run: poetry run python scripts/test_llm_transformers.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
