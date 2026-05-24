"""Command-line interface for brain tumor MRI model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training import SUPPORTED_ARCHITECTURES, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a brain tumor MRI classifier.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root containing Training/Testing.")
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--scheduler", choices=["reduce_on_plateau", "cosine", "none"], default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ImageNet weights.")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Train the full backbone.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    parser.add_argument("--dry-run", action="store_true", help="Build loaders/model and run one forward pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "dataset_root": args.dataset_root,
        "architecture": args.architecture,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "val_size": args.val_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "dropout": args.dropout,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "scheduler": args.scheduler,
        "device": args.device,
        "output_dir": args.output_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "run_name": args.run_name,
        "pretrained": False if args.no_pretrained else None,
        "freeze_backbone": False if args.unfreeze_backbone else None,
        "use_amp": False if args.no_amp else None,
        "tensorboard": False if args.no_tensorboard else None,
        "dry_run": True if args.dry_run else None,
    }
    run_training(config_path=args.config, **overrides)
