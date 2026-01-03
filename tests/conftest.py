"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

from ahmedkobtan_cinema_robot_playground.src.models.appconfig import AppConfig
from ahmedkobtan_cinema_robot_playground.src.utils.config_utils import parse_config


@pytest.fixture
def ip_webcam_url():
    """
    Fixture to get IP webcam URL from config.

    Returns:
        IP webcam URL string from config file.
    """
    # Get environment (default to 'dev')
    env = os.getenv("ENV", "dev").lower()

    # Get project root (3 levels up from tests/)
    project_root = Path(__file__).parent.parent
    config_file = project_root / "config" / f"{env}.json"

    # Fallback to dev.json if specified env file doesn't exist
    if not config_file.exists():
        config_file = project_root / "config" / "dev.json"

    # Parse and validate config
    config_data = parse_config(str(config_file))
    app_config = AppConfig(**config_data)

    return app_config.configResolution.resolved.ip_webcam_url
