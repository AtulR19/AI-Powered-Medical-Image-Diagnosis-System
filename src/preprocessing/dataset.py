"""Dataset abstractions for medical image classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageRecord:
    """Single image sample discovered from an image-folder dataset."""

    path: Path
    label: int
    label_name: str


def discover_classes(split_dir: str | Path) -> list[str]:
    """Return sorted class folder names from a split directory."""

    root = Path(split_dir)
    if not root.exists():
        raise FileNotFoundError(f"Split directory does not exist: {root}")

    class_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class directories found in: {root}")

    return class_names


def build_class_to_idx(split_dir: str | Path) -> dict[str, int]:
    """Build a deterministic class-to-index mapping from class folders."""

    return {class_name: index for index, class_name in enumerate(discover_classes(split_dir))}


def discover_image_records(
    split_dir: str | Path,
    class_to_idx: Mapping[str, int] | None = None,
) -> list[ImageRecord]:
    """Discover image paths and labels from a split directory.

    The expected layout is:

    split_dir/
        glioma/
        meningioma/
        notumor/
        pituitary/
    """

    root = Path(split_dir)
    if class_to_idx is None:
        class_to_idx = build_class_to_idx(root)

    known_classes = set(class_to_idx)
    discovered_classes = set(discover_classes(root))
    unknown_classes = discovered_classes - known_classes
    if unknown_classes:
        unknown = ", ".join(sorted(unknown_classes))
        raise ValueError(f"Found classes not present in class_to_idx for {root}: {unknown}")

    records: list[ImageRecord] = []
    for class_name in sorted(class_to_idx):
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        label = class_to_idx[class_name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                records.append(ImageRecord(path=image_path, label=label, label_name=class_name))

    if not records:
        raise ValueError(f"No supported image files found in: {root}")

    return records


def read_rgb_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk as an RGB NumPy array."""

    path = Path(image_path)
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


class BrainTumorMRIDataset(Dataset):
    """PyTorch dataset for Brain Tumor MRI image-folder records."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        if not records:
            raise ValueError("BrainTumorMRIDataset requires at least one image record.")

        self.records = list(records)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = read_rgb_image(record.path)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0)

        label = torch.tensor(record.label, dtype=torch.long)

        if self.return_path:
            return image, label, str(record.path)

        return image, label

    @property
    def labels(self) -> list[int]:
        """Return numeric labels for stratified splitting or diagnostics."""

        return [record.label for record in self.records]

    @property
    def paths(self) -> list[Path]:
        """Return image paths in dataset order."""

        return [record.path for record in self.records]


class MedicalImageDataset(Dataset):
    """Load image paths and labels from a metadata CSV file."""

    def __init__(
        self,
        metadata_csv: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
        image_column: str = "image_path",
        label_column: str = "label",
    ) -> None:
        self.metadata = pd.read_csv(metadata_csv)
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self.image_root / row[self.image_column]
        image = Image.open(image_path).convert("RGB")
        label = row[self.label_column]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
