"""Inference pipeline for trained brain tumor MRI classifiers."""

from __future__ import annotations

import io
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.preprocessing.transforms import build_eval_transforms
from src.training.model_factory import create_model


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction response for one MRI image."""

    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_device(requested_device: str = "auto") -> torch.device:
    """Resolve an inference device from a user-facing string."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def load_rgb_image(image_source: str | Path | bytes | BinaryIO | Image.Image) -> Image.Image:
    """Load an uploaded image, byte stream, path, or PIL image as RGB."""

    try:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, bytes):
            return Image.open(io.BytesIO(image_source)).convert("RGB")
        if hasattr(image_source, "read"):
            return Image.open(image_source).convert("RGB")
        return Image.open(Path(image_source)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Could not read a valid image file.") from exc


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """Load a project checkpoint with safer torch defaults when available."""

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


class BrainTumorPredictor:
    """Load a trained checkpoint and predict brain tumor classes from MRI images."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        image_size: int | None = None,
        architecture: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = resolve_device(device)
        self.checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        self.config = self.checkpoint.get("config", {})
        self.class_to_idx = self._load_class_mapping()
        self.idx_to_class = {index: class_name for class_name, index in self.class_to_idx.items()}
        self.class_names = [self.idx_to_class[index] for index in sorted(self.idx_to_class)]
        self.image_size = image_size or int(self.config.get("image_size", 224))
        self.architecture = architecture or self.config.get("architecture", "resnet50")
        self.transform = build_eval_transforms(image_size=self.image_size)
        self.model = self._load_model()

    def _load_class_mapping(self) -> dict[str, int]:
        class_to_idx = self.checkpoint.get("class_to_idx")
        if not class_to_idx:
            raise ValueError("Checkpoint is missing 'class_to_idx'.")
        return {str(class_name): int(index) for class_name, index in class_to_idx.items()}

    def _load_model(self) -> torch.nn.Module:
        model = create_model(
            architecture=self.architecture,
            num_classes=len(self.class_to_idx),
            pretrained=False,
            freeze_backbone=False,
            dropout=float(self.config.get("dropout", 0.2)),
        )
        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image_source: str | Path | bytes | BinaryIO | Image.Image) -> torch.Tensor:
        """Preprocess an image into a normalized model input tensor."""

        image = load_rgb_image(image_source)
        image_array = np.array(image, dtype=np.uint8)
        tensor = self.transform(image=image_array)["image"]
        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, image_source: str | Path | bytes | BinaryIO | Image.Image) -> PredictionResult:
        """Predict the most likely class and probabilities for one image."""

        tensor = self.preprocess(image_source)
        logits = self.model(tensor)
        probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
        predicted_index = int(torch.argmax(probabilities_tensor).item())
        predicted_class = self.idx_to_class[predicted_index]
        probabilities = {
            self.idx_to_class[index]: float(probabilities_tensor[index].item())
            for index in range(len(self.class_names))
        }

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            confidence=probabilities[predicted_class],
            probabilities=probabilities,
        )


class MedicalImagePredictor(BrainTumorPredictor):
    """Backward-compatible alias for the brain tumor inference pipeline."""


if __name__ == "__main__":
    from predict import main

    main()
