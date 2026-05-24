"""Grad-CAM visualization for trained brain tumor classifiers."""

from __future__ import annotations

import json
import os
import sys
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.inference import BrainTumorPredictor
from src.inference.predictor import load_rgb_image


@dataclass(frozen=True)
class GradCAMOutput:
    """Paths and prediction metadata produced by Grad-CAM generation."""

    predicted_class: str
    target_class: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_path: str
    overlay_path: str
    confidence_path: str
    summary_path: str
    metadata_path: str

    def to_dict(self) -> dict:
        return asdict(self)


class GradCAM:
    """Generate Grad-CAM heatmaps from a model and target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        """Remove registered hooks."""

        self.forward_handle.remove()
        self.backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a normalized heatmap and return model logits."""

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)

        cam_min = cam.min()
        cam_max = cam.max()
        heatmap = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return heatmap.detach().cpu(), logits.detach().cpu()


def get_default_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return a sensible final convolutional layer for supported architectures."""

    architecture = architecture.lower()
    if architecture == "resnet50":
        return model.layer4[-1]
    if architecture.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"Unsupported architecture for Grad-CAM: {architecture}")


def colorize_heatmap(heatmap: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """Convert a 0-1 heatmap into an RGB color image."""

    colormap = plt.get_cmap(cmap_name)
    colored = colormap(heatmap)[..., :3]
    return (colored * 255).astype(np.uint8)


def make_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    cmap_name: str = "jet",
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on top of an RGB image."""

    base = np.asarray(image).astype(np.float32)
    colored_heatmap = colorize_heatmap(heatmap, cmap_name=cmap_name).astype(np.float32)
    overlay = (1.0 - alpha) * base + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_confidence_chart(
    probabilities: dict[str, float],
    predicted_class: str,
    output_path: str | Path,
) -> None:
    """Save a horizontal confidence/probability chart."""

    class_names = list(probabilities.keys())
    values = [probabilities[class_name] for class_name in class_names]
    colors = ["#2f80ed" if class_name == predicted_class else "#8c98a8" for class_name in class_names]

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(class_names, values, color=colors)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Probability")
    axis.set_title(f"Prediction Confidence: {predicted_class} ({probabilities[predicted_class]:.2%})")
    axis.invert_yaxis()

    for index, value in enumerate(values):
        axis.text(min(value + 0.02, 0.98), index, f"{value:.2%}", va="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_summary_figure(
    original: Image.Image,
    heatmap_image: Image.Image,
    overlay: Image.Image,
    probabilities: dict[str, float],
    predicted_class: str,
    target_class: str,
    output_path: str | Path,
) -> None:
    """Save a single figure containing original, heatmap, overlay, and confidence."""

    class_names = list(probabilities.keys())
    values = [probabilities[class_name] for class_name in class_names]
    colors = ["#2f80ed" if class_name == predicted_class else "#8c98a8" for class_name in class_names]

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Preprocessed MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(heatmap_image)
    axes[0, 1].set_title(f"Grad-CAM Heatmap: {target_class}")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Heatmap Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].barh(class_names, values, color=colors)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Probability")
    axes[1, 1].set_title(f"Predicted: {predicted_class} ({probabilities[predicted_class]:.2%})")
    for index, value in enumerate(values):
        axes[1, 1].text(min(value + 0.02, 0.98), index, f"{value:.2%}", va="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int | None = None,
) -> torch.Tensor:
    """Generate a Grad-CAM heatmap for a model prediction."""

    model.eval()
    with GradCAM(model, target_layer) as gradcam:
        with torch.enable_grad():
            logits = model(input_tensor)
            resolved_target = int(torch.argmax(logits, dim=1).item()) if target_class is None else target_class
            model.zero_grad(set_to_none=True)
            heatmap, _ = gradcam.generate(input_tensor, resolved_target)
    return heatmap


def explain_image(
    checkpoint_path: str | Path,
    image_path: str | Path | bytes | BinaryIO | Image.Image,
    output_dir: str | Path = "outputs/gradcam",
    device: str = "auto",
    target_class: str | int | None = None,
    alpha: float = 0.4,
    image_size: int | None = None,
) -> GradCAMOutput:
    """Run prediction plus Grad-CAM and save visualization artifacts."""

    predictor = BrainTumorPredictor(
        checkpoint_path=checkpoint_path,
        device=device,
        image_size=image_size,
    )
    return explain_with_predictor(
        predictor=predictor,
        image_path=image_path,
        output_dir=output_dir,
        target_class=target_class,
        alpha=alpha,
    )


def explain_with_predictor(
    predictor: BrainTumorPredictor,
    image_path: str | Path | bytes | BinaryIO | Image.Image,
    output_dir: str | Path = "outputs/gradcam",
    target_class: str | int | None = None,
    alpha: float = 0.4,
    output_stem: str | None = None,
) -> GradCAMOutput:
    """Run prediction plus Grad-CAM using an already-loaded predictor."""

    prediction = predictor.predict(image_path)

    if target_class is None:
        target_index = prediction.predicted_index
    elif isinstance(target_class, str):
        if target_class not in predictor.class_to_idx:
            labels = ", ".join(predictor.class_names)
            raise ValueError(f"Unknown target class '{target_class}'. Choose one of: {labels}")
        target_index = predictor.class_to_idx[target_class]
    else:
        target_index = int(target_class)

    target_name = predictor.idx_to_class[target_index]
    input_tensor = predictor.preprocess(image_path)
    target_layer = get_default_target_layer(predictor.model, predictor.architecture)

    predictor.model.eval()
    with GradCAM(predictor.model, target_layer) as gradcam:
        with torch.enable_grad():
            heatmap_tensor, _ = gradcam.generate(input_tensor, target_index)

    original = load_rgb_image(image_path).resize(
        (predictor.image_size, predictor.image_size),
        resample=Image.BILINEAR,
    )
    heatmap = heatmap_tensor.numpy()
    heatmap_image = Image.fromarray(colorize_heatmap(heatmap))
    overlay = make_overlay(original, heatmap, alpha=alpha)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = output_stem or (Path(image_path).stem if isinstance(image_path, (str, Path)) else "uploaded_image")

    heatmap_path = output_path / f"{stem}_gradcam_heatmap.png"
    overlay_path = output_path / f"{stem}_gradcam_overlay.png"
    confidence_path = output_path / f"{stem}_confidence.png"
    summary_path = output_path / f"{stem}_gradcam_summary.png"
    metadata_path = output_path / f"{stem}_gradcam_metadata.json"

    heatmap_image.save(heatmap_path)
    overlay.save(overlay_path)
    save_confidence_chart(prediction.probabilities, prediction.predicted_class, confidence_path)
    save_summary_figure(
        original=original,
        heatmap_image=heatmap_image,
        overlay=overlay,
        probabilities=prediction.probabilities,
        predicted_class=prediction.predicted_class,
        target_class=target_name,
        output_path=summary_path,
    )

    result = GradCAMOutput(
        predicted_class=prediction.predicted_class,
        target_class=target_name,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        heatmap_path=str(heatmap_path),
        overlay_path=str(overlay_path),
        confidence_path=str(confidence_path),
        summary_path=str(summary_path),
        metadata_path=str(metadata_path),
    )
    metadata_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for an MRI image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or latest.pt.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an MRI image.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam"), help="Directory for outputs.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--target-class", default=None, help="Optional class name or class index to explain.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Heatmap overlay opacity.")
    parser.add_argument("--image-size", type=int, default=None, help="Optional preprocessing image size override.")
    return parser.parse_args()


def parse_target_class(value: str | None):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = parse_args()
    result = explain_image(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        output_dir=args.output_dir,
        device=args.device,
        target_class=parse_target_class(args.target_class),
        alpha=args.alpha,
        image_size=args.image_size,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
