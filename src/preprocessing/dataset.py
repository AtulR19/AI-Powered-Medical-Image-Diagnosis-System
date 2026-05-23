"""Dataset abstractions for medical image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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
