"""Director Agent (Layer 1) - Command parsing and object grounding."""

import re
from typing import Dict, Optional

from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.models.detection_models import (
    BoundingBox,
    DetectionModel,
    GroundingDINOModel,
    MockDetectionModel,
)


class DirectorAgent:
    """Director Agent for parsing commands and grounding objects."""

    def __init__(
        self, detection_model: Optional[DetectionModel] = None, use_llm: bool = False
    ):
        """
        Initialize Director Agent.

        Args:
            detection_model: Detection model to use (None for auto-select)
            use_llm: Whether to use LLM for command parsing (requires Ollama)
        """
        self.detection_model = detection_model or self._create_detection_model()
        self.use_llm = use_llm
        self.llm_agent = None

        if use_llm:
            self._setup_llm_agent()

    def _create_detection_model(self) -> DetectionModel:
        """Create detection model (try real model, fallback to mock)."""
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("Creating Grounding DINO model")
                return GroundingDINOModel()
            else:
                logger.warning("CUDA not available, using mock detection model")
                return MockDetectionModel()
        except Exception as e:
            logger.warning(f"Error creating detection model: {e}, using mock")
            return MockDetectionModel()

    def _setup_llm_agent(self) -> None:
        """Set up LLM agent for command parsing."""
        try:
            from langchain_community.llms import Ollama

            logger.info("Setting up LangChain ReAct agent with Ollama")
            llm = Ollama(model="llama3.1:8b")

            # Simple agent (can be enhanced with tools later)
            self.llm_agent = llm

        except Exception as e:
            logger.warning(f"Error setting up LLM agent: {e}, using simple parsing")
            self.use_llm = False

    def parse_command(self, command: str) -> Dict[str, str]:
        """
        Parse user command to extract object and shot type.

        Args:
            command: User command string (e.g., "Orbit the red cup")

        Returns:
            Dictionary with "object" and "shot_type" keys
        """
        if self.use_llm and self.llm_agent is not None:
            return self._parse_with_llm(command)
        else:
            return self._parse_simple(command)

    def _parse_with_llm(self, command: str) -> Dict[str, str]:
        """Parse command using LLM."""
        try:
            prompt = f"""Parse this cinema robot command: "{command}"

Extract:
1. Object to track
2. Shot type (follow, dolly_in, dolly_out, orbit, pan)

Respond in JSON: {{"object": "...", "shot_type": "..."}}"""

            response = self.llm_agent.invoke(prompt)
            # Simple JSON extraction (can be improved)
            import json

            # Try to extract JSON from response
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(
                f"Error parsing with LLM: {e}, falling back to simple parsing"
            )

        return self._parse_simple(command)

    def _parse_simple(self, command: str) -> Dict[str, str]:
        """Simple rule-based command parsing."""
        command_lower = command.lower()

        # Extract shot type
        shot_type = "follow"  # default
        if "orbit" in command_lower:
            shot_type = "orbit"
        elif "dolly in" in command_lower or "dollyin" in command_lower:
            shot_type = "dolly_in"
        elif "dolly out" in command_lower or "dollyout" in command_lower:
            shot_type = "dolly_out"
        elif "pan" in command_lower:
            shot_type = "pan"
        elif "follow" in command_lower or "track" in command_lower:
            shot_type = "follow"

        # Extract object (remove shot type keywords)
        object_keywords = ["orbit", "dolly", "pan", "follow", "track", "the", "a", "an"]
        words = command_lower.split()
        object_words = [w for w in words if w not in object_keywords]
        object_name = " ".join(object_words) if object_words else "object"

        return {"object": object_name, "shot_type": shot_type}

    def ground_object(self, image, text_prompt: str) -> Optional[BoundingBox]:
        """
        Ground object in image using detection model.

        Args:
            image: Input image (numpy array, BGR format)
            text_prompt: Text description of object

        Returns:
            Bounding box or None if not found
        """
        try:
            boxes = self.detection_model.detect(image, text_prompt=text_prompt)
            if boxes:
                # Return highest confidence box
                best_box = max(boxes, key=lambda b: b.confidence)
                logger.info(f"Object grounded: {best_box.to_dict()}")
                return best_box
            else:
                logger.warning(f"No objects found for prompt: {text_prompt}")
                return None
        except Exception as e:
            logger.error(f"Error grounding object: {e}")
            return None

    def process_command(self, command: str, image) -> Optional[Dict]:
        """
        Process user command: parse and ground object.

        Args:
            command: User command string
            image: Current video frame

        Returns:
            Dictionary with object info and shot type, or None if failed
        """
        # Parse command
        parsed = self.parse_command(command)
        object_name = parsed.get("object", "object")
        shot_type = parsed.get("shot_type", "follow")

        # Ground object
        bbox = self.ground_object(image, object_name)
        if bbox is None:
            return None

        return {
            "object_name": object_name,
            "shot_type": shot_type,
            "bounding_box": bbox,
        }
