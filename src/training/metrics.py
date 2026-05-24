"""Metrics and reporting helpers for classification training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and accuracy values for one epoch/split."""

    loss: float
    accuracy: float


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Return correct predictions and sample count."""

    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct, total


def save_history(history: list[dict], output_path: str | Path) -> None:
    """Save epoch history to CSV."""

    pd.DataFrame(history).to_csv(output_path, index=False)


def save_json(data: dict, output_path: str | Path) -> None:
    """Save a JSON file with stable formatting."""

    path = Path(output_path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_classification_outputs(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_dir: str | Path,
    prefix: str = "test",
) -> dict:
    """Save confusion matrix and classification report artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    np.savetxt(output_path / f"{prefix}_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    save_json(report_dict, output_path / f"{prefix}_classification_report.json")
    (output_path / f"{prefix}_classification_report.txt").write_text(report_text, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title(f"{prefix.title()} Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(labels)
    axis.set_yticks(labels)
    axis.set_xticklabels(class_names, rotation=30, ha="right")
    axis.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            axis.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path / f"{prefix}_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    return {
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
    }


def epoch_metrics_to_dict(metrics: EpochMetrics) -> dict[str, float]:
    """Convert epoch metrics to serializable values."""

    return asdict(metrics)
