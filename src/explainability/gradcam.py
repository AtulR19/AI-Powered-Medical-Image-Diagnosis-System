"""Grad-CAM integration points."""

from __future__ import annotations

import torch


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int | None = None,
) -> torch.Tensor:
    """Generate a Grad-CAM heatmap for a model prediction."""

    raise NotImplementedError("Connect Captum or a Grad-CAM implementation here.")
