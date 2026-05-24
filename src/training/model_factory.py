"""Model factory for transfer-learning backbones."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchvision import models


SUPPORTED_ARCHITECTURES = (
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _build_with_weights(
    builder: Callable,
    weights_cls,
    pretrained: bool,
) -> nn.Module:
    weights = weights_cls.DEFAULT if pretrained else None
    try:
        return builder(weights=weights)
    except TypeError:
        return builder(pretrained=pretrained)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Could not load pretrained weights. Check your internet connection, "
                "or run with --no-pretrained for an offline smoke test."
            ) from exc
        raise


def _create_resnet50(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    model = _build_with_weights(models.resnet50, models.ResNet50_Weights, pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _create_efficientnet(
    architecture: str,
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    builders = {
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights),
    }
    builder, weights_cls = builders[architecture]
    model = _build_with_weights(builder, weights_cls, pretrained=pretrained)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def create_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a transfer-learning classifier model."""

    architecture = architecture.lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(f"Unsupported architecture '{architecture}'. Choose one of: {supported}")

    if architecture == "resnet50":
        model = _create_resnet50(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
        head = model.fc
    else:
        model = _create_efficientnet(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
        head = model.classifier

    if freeze_backbone:
        _set_trainable(model, trainable=False)
        _set_trainable(head, trainable=True)

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable model parameters."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def move_batch_to_device(batch, device: torch.device):
    """Move a DataLoader batch to the selected device."""

    if len(batch) == 3:
        images, labels, paths = batch
        return images.to(device, non_blocking=True), labels.to(device, non_blocking=True), paths

    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True), None
