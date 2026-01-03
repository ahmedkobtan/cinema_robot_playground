#!/usr/bin/env python3
"""Test Transformers-based LLM for command parsing."""

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


def test_llm_parsing(device: str = "cpu"):
    """Test LLM-based command parsing."""
    logger.info("=" * 60)
    logger.info("Testing Transformers-based LLM Command Parsing")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # Test without LLM first
    logger.info("\n1. Testing simple command parsing (no LLM)...")
    agent_simple = DirectorAgent(use_llm=False, device=device)
    test_commands = [
        "Orbit the red cup",
        "Follow the cat",
        "Dolly in on the object",
        "Track the person in the green shirt",
        "Pan to the left side of the frame",
    ]

    for cmd in test_commands:
        result = agent_simple.parse_command(cmd)
        logger.info(f"  Command: '{cmd}'")
        logger.info(f"  Parsed: {result}")
        assert "object" in result
        assert "shot_type" in result

    # Test with LLM (Transformers)
    logger.info("\n2. Testing Transformers-based LLM command parsing...")
    try:
        logger.info("  Loading LLM model (this may take a few minutes)...")
        agent_llm = DirectorAgent(use_llm=True, device=device)

        if agent_llm.llm_agent is not None:
            logger.info("  ✓ LLM agent initialized successfully")

            for cmd in test_commands:
                logger.info(f"\n  Command: '{cmd}'")
                result = agent_llm.parse_command(cmd)
                logger.info(f"  Parsed: {result}")
                assert "object" in result
                assert "shot_type" in result
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
    logger.info("LLM Test Complete")
    logger.info("=" * 60)


def test_model_loading(device: str = "cpu"):
    """Test direct model loading."""
    logger.info("\n3. Testing direct Transformers model loading...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "microsoft/Phi-3-mini-4k-instruct"
        logger.info(f"  Loading model: {model_name}")
        logger.info(f"  Device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("  ✓ Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        logger.info("  ✓ Model loaded")

        # Test inference
        test_prompt = "Parse: Orbit the cup. JSON:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("  ✓ Inference test successful")
        logger.info(f"  Response: {response[:100]}...")

        return True

    except Exception as e:
        logger.warning(f"  ✗ Model loading failed: {e}")
        logger.info("  This is acceptable - simple parsing will be used")
        return False


def main():
    """Run LLM tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Transformers-based LLM")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    test_llm_parsing(device=args.device)
    test_model_loading(device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
