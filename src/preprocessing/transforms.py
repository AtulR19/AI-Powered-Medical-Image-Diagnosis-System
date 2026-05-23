"""Image transformation pipelines."""

from __future__ import annotations

from torchvision import transforms


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """Create standard transforms for training or evaluation."""

    steps = [
        transforms.Resize((image_size, image_size)),
    ]

    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return transforms.Compose(steps)
