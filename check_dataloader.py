"""Smoke-test the Brain Tumor MRI PyTorch DataLoader pipeline."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    print("Starting Brain Tumor MRI DataLoader check...", flush=True)
    print("Importing preprocessing pipeline...", flush=True)

    from src.preprocessing import create_brain_tumor_dataloaders

    dataset_root = Path("data/archive (1)")
    print(f"Dataset root: {dataset_root.resolve()}", flush=True)
    print("Creating train/validation/test loaders...", flush=True)

    loaders = create_brain_tumor_dataloaders(
        dataset_root=dataset_root,
        image_size=224,
        batch_size=8,
        val_size=0.2,
        num_workers=0,
        seed=42,
    )

    print(f"Classes: {loaders.class_to_idx}", flush=True)
    print(f"Split sizes: {loaders.split_sizes}", flush=True)
    print("Loading one training batch...", flush=True)

    images, labels = next(iter(loaders.train))

    print(f"Image batch shape: {images.shape}", flush=True)
    print(f"Label batch shape: {labels.shape}", flush=True)
    print(f"Example labels: {labels.tolist()}", flush=True)
    print("DataLoader check passed.", flush=True)


if __name__ == "__main__":
    main()
