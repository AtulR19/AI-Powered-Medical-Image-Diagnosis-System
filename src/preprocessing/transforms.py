"""Albumentations image transformation pipelines."""

from __future__ import annotations

from typing import Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> A.Compose:
    """Create augmentation and normalization transforms for training images."""

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-10, 10),
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.12,
                p=0.4,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> A.Compose:
    """Create deterministic resizing and normalization transforms for validation/test."""

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_transforms(
    image_size: int = 224,
    train: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> A.Compose:
    """Create transforms for either train or evaluation mode."""

    if train:
        return build_train_transforms(image_size=image_size, mean=mean, std=std)
    return build_eval_transforms(image_size=image_size, mean=mean, std=std)
