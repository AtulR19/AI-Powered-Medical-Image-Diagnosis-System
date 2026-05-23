"""Reusable inference helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


class MedicalImagePredictor:
    """Wrap model loading and prediction for a trained PyTorch model."""

    def __init__(self, model: torch.nn.Module, transform, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transform

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(tensor)
