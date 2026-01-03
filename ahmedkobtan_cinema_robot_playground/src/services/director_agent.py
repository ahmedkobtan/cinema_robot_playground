"""Director Agent (Layer 1) - Command parsing and object grounding."""

import re
from typing import Dict, Optional

import torch
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
        self,
        detection_model: Optional[DetectionModel] = None,
        use_llm: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize Director Agent.

        Args:
            detection_model: Detection model to use (None for auto-select)
            use_llm: Whether to use LLM for command parsing (uses Transformers)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_model = detection_model or self._create_detection_model()
        self.use_llm = use_llm
        self.llm_agent = None
        self.llm_model = None
        self.llm_tokenizer = None

        if use_llm:
            self._setup_llm_agent()

    def _create_detection_model(self) -> DetectionModel:
        """Create detection model (try real model, fallback to mock)."""
        try:
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Creating Grounding DINO model on {device}")
            return GroundingDINOModel(device=device)
        except Exception as e:
            logger.warning(f"Error creating detection model: {e}, using mock")
            return MockDetectionModel()

    def _setup_llm_agent(self) -> None:
        """Set up LLM agent for command parsing using Transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            # Use a small, efficient model that fits on RTX 2080 Ti
            # Phi-3-mini is a good choice: 3.8B parameters, runs well on 11GB VRAM
            model_name = "microsoft/Phi-3-mini-4k-instruct"

            logger.info(f"Loading LLM model: {model_name}")
            logger.info("This may take a few minutes on first run...")

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            device_map = "auto" if device == "cuda" else None

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device_map,
                trust_remote_code=True,
            )

            # Create pipeline
            self.llm_agent = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                max_new_tokens=100,
                do_sample=False,  # Deterministic output
            )

            logger.info("LLM agent loaded successfully")

        except Exception as e:
            logger.warning(f"Error setting up LLM agent: {e}, using simple parsing")
            logger.info("Falling back to simple rule-based parsing")
            self.use_llm = False
            self.llm_agent = None

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
        """Parse command using LLM (Transformers)."""
        try:
            prompt = f"""<|user|>
Parse this cinema robot command: "{command}"

Extract:
1. Object to track (the thing to follow)
2. Shot type (one of: follow, dolly_in, dolly_out, orbit, pan)

Respond ONLY with valid JSON: {{"object": "object_name", "shot_type": "shot_type"}}
<|assistant|>
"""

            # Generate response
            results = self.llm_agent(
                prompt,
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False,
            )

            # Extract generated text
            if isinstance(results, list) and len(results) > 0:
                response = results[0].get("generated_text", "")
                # Remove the prompt from response
                if prompt in response:
                    response = response[len(prompt) :].strip()
            else:
                response = str(results)

            # Extract JSON from response
            import json

            # Try to find JSON object
            json_match = re.search(
                r"\{[^{}]*\"object\"[^{}]*\"shot_type\"[^{}]*\}", response
            )
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate and return
                if "object" in parsed and "shot_type" in parsed:
                    return {
                        "object": str(parsed["object"]).strip(),
                        "shot_type": str(parsed["shot_type"]).strip().lower(),
                    }

            # If JSON extraction failed, try simple parsing
            logger.warning(
                "LLM response did not contain valid JSON, using simple parsing"
            )
            return self._parse_simple(command)

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
