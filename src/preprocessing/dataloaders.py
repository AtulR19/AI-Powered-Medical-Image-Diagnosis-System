"""DataLoader factory for the Brain Tumor MRI dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.preprocessing.dataset import (
    BrainTumorMRIDataset,
    ImageRecord,
    build_class_to_idx,
    discover_image_records,
)
from src.preprocessing.transforms import build_eval_transforms, build_train_transforms


@dataclass(frozen=True)
class BrainTumorDataLoaders:
    """Container returned by the Brain Tumor MRI DataLoader factory."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    class_names: list[str]
    class_to_idx: dict[str, int]
    split_sizes: dict[str, int]

    @property
    def val(self) -> DataLoader:
        """Alias for the validation DataLoader."""

        return self.validation

    def as_dict(self) -> dict[str, DataLoader]:
        """Return loaders in a plain dictionary for training loops."""

        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def split_train_validation_records(
    records: list[ImageRecord],
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Create a stratified train/validation split from training records."""

    labels = [record.label for record in records]
    train_records, validation_records = train_test_split(
        records,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    return list(train_records), list(validation_records)


def create_brain_tumor_dataloaders(
    dataset_root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    val_size: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    return_paths: bool = False,
    drop_last: bool = False,
) -> BrainTumorDataLoaders:
    """Build train, validation, and test DataLoaders for Brain Tumor MRI images.

    The expected dataset layout is:

    dataset_root/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/
    """

    root = Path(dataset_root)
    training_dir = root / "Training"
    testing_dir = root / "Testing"

    if not training_dir.exists():
        raise FileNotFoundError(f"Training split not found: {training_dir}")
    if not testing_dir.exists():
        raise FileNotFoundError(f"Testing split not found: {testing_dir}")

    class_to_idx = build_class_to_idx(training_dir)
    class_names = [class_name for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    training_records = discover_image_records(training_dir, class_to_idx=class_to_idx)
    test_records = discover_image_records(testing_dir, class_to_idx=class_to_idx)
    train_records, validation_records = split_train_validation_records(
        records=training_records,
        val_size=val_size,
        seed=seed,
    )

    train_dataset = BrainTumorMRIDataset(
        records=train_records,
        transform=build_train_transforms(image_size=image_size),
        return_path=return_paths,
    )
    validation_dataset = BrainTumorMRIDataset(
        records=validation_records,
        transform=build_eval_transforms(image_size=image_size),
        return_path=return_paths,
    )
    test_dataset = BrainTumorMRIDataset(
        records=test_records,
        transform=build_eval_transforms(image_size=image_size),
        return_path=return_paths,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator().manual_seed(seed)
    common_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
        **common_loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    return BrainTumorDataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        class_names=class_names,
        class_to_idx=dict(class_to_idx),
        split_sizes={
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
    )
