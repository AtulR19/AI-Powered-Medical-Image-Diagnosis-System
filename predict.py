"""Command-line inference for brain tumor MRI images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference import BrainTumorPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict brain tumor class from an MRI image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an MRI image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or latest.pt.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--image-size", type=int, default=None, help="Optional preprocessing image size override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = BrainTumorPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
    )
    result = predictor.predict(args.image)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
