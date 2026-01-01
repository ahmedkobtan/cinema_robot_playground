#!/usr/bin/env python3
"""Test LangChain + Ollama setup for Director Agent."""

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


def test_llm_agent():
    """Test LLM agent setup and command parsing."""
    logger.info("=" * 60)
    logger.info("Testing LangChain + Ollama Setup")
    logger.info("=" * 60)

    # Test without LLM (simple parsing)
    logger.info("\n1. Testing simple command parsing (no LLM)...")
    agent_simple = DirectorAgent(use_llm=False)
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

    # Test with LLM (if available)
    logger.info("\n2. Testing LLM-based command parsing...")
    try:
        agent_llm = DirectorAgent(use_llm=True)
        if agent_llm.llm_agent is not None:
            logger.info("  ✓ LLM agent initialized successfully")

            for cmd in test_commands:
                result = agent_llm.parse_command(cmd)
                logger.info(f"  Command: '{cmd}'")
                logger.info(f"  Parsed: {result}")
        else:
            logger.warning("  ✗ LLM agent not available (Ollama may not be running)")
            logger.info("  To use LLM parsing:")
            logger.info("    1. Install Ollama: https://ollama.ai")
            logger.info("    2. Pull model: ollama pull llama3.1:8b")
            logger.info("    3. Start Ollama service")

    except Exception as e:
        logger.error(f"  ✗ Error setting up LLM agent: {e}")
        logger.info("  Falling back to simple parsing")

    logger.info("\n" + "=" * 60)
    logger.info("LLM Agent Test Complete")
    logger.info("=" * 60)


def test_ollama_connection():
    """Test direct Ollama connection."""
    logger.info("\n3. Testing direct Ollama connection...")
    try:
        from langchain_community.llms import Ollama

        llm = Ollama(model="llama3.1:8b")
        response = llm.invoke("Say 'Hello' if you can hear me.")
        logger.info(f"  ✓ Ollama connected: {response[:50]}...")
        return True
    except Exception as e:
        logger.warning(f"  ✗ Ollama not available: {e}")
        logger.info("  Install and start Ollama to enable LLM parsing")
        return False


def main():
    """Run LLM agent tests."""
    test_llm_agent()
    test_ollama_connection()
    return 0


if __name__ == "__main__":
    sys.exit(main())
