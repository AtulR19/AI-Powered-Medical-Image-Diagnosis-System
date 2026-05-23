"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    import yaml

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in config file: {path}")

    return config
