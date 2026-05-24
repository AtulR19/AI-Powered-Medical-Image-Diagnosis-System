"""Export a training checkpoint as a smaller inference checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def load_training_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint on CPU with compatibility across PyTorch versions."""

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def default_output_path(checkpoint_path: Path) -> Path:
    """Build a default export path next to the model exports folder."""

    run_name = checkpoint_path.parent.name if checkpoint_path.parent.name else checkpoint_path.stem
    return Path("models") / "exports" / f"{run_name}-inference.pt"


def export_inference_checkpoint(checkpoint_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Save only the data needed by the inference and Grad-CAM pipelines."""

    source_path = Path(checkpoint_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source_path}")

    checkpoint = load_training_checkpoint(source_path)
    required_keys = ("model_state_dict", "class_to_idx")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(f"Checkpoint is missing required keys: {', '.join(missing_keys)}")

    destination_path = Path(output_path) if output_path else default_output_path(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    inference_checkpoint = {
        "model_state_dict": checkpoint["model_state_dict"],
        "class_to_idx": checkpoint["class_to_idx"],
        "config": checkpoint.get("config", {}),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "epoch": checkpoint.get("epoch"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_path),
        "checkpoint_type": "inference",
    }
    torch.save(inference_checkpoint, destination_path)
    return destination_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained checkpoint for deployment inference.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or latest.pt.")
    parser.add_argument("--output", type=Path, default=None, help="Output .pt path. Defaults to models/exports/<run>-inference.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_inference_checkpoint(args.checkpoint, args.output)
    print(f"Exported inference checkpoint: {output_path}")


if __name__ == "__main__":
    main()
