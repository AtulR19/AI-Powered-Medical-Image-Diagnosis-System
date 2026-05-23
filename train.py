"""Training entry point for the medical image diagnosis system."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a medical image diagnosis model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML training configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(config_path=args.config)


if __name__ == "__main__":
    main()
