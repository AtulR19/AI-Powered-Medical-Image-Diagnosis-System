"""Training orchestration for PyTorch models."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training(config_path: str | Path | None = None) -> None:
    """Run model training from an optional YAML configuration file."""

    config = load_config(config_path) if config_path else {}
    logger.info("Training pipeline initialized.")

    if config:
        logger.info("Loaded training config with keys: %s", ", ".join(sorted(config.keys())))
    else:
        logger.info("No config file supplied. Add dataset/model settings before training.")

    logger.info("Scaffold ready: connect datasets, model definition, optimizer, and metrics here.")
