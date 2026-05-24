"""Complete PyTorch training pipeline for brain tumor classification."""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm

from src.preprocessing import create_brain_tumor_dataloaders
from src.training.early_stopping import EarlyStopping
from src.training.metrics import (
    EpochMetrics,
    accuracy_from_logits,
    epoch_metrics_to_dict,
    save_classification_outputs,
    save_history,
    save_json,
)
from src.training.model_factory import count_parameters, create_model, move_batch_to_device
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the brain tumor training pipeline."""

    dataset_root: str = "data/archive (1)"
    architecture: str = "resnet50"
    image_size: int = 224
    batch_size: int = 32
    val_size: float = 0.2
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.2
    patience: int = 5
    min_delta: float = 0.0
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    device: str = "auto"
    use_amp: bool = True
    output_dir: str = "outputs/training"
    checkpoint_dir: str = "models/checkpoints"
    run_name: str | None = None
    tensorboard: bool = True
    dry_run: bool = False


class NullSummaryWriter:
    """No-op writer used when TensorBoard is unavailable or disabled."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def add_text(self, *args, **kwargs) -> None:
        return None

    def add_figure(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def create_summary_writer(log_dir: Path, enabled: bool = True):
    """Create a TensorBoard SummaryWriter with a graceful fallback."""

    if not enabled:
        return NullSummaryWriter()

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        logger.warning("TensorBoard is not installed. Install 'tensorboard' to enable event logging.")
        return NullSummaryWriter()

    return SummaryWriter(log_dir=str(log_dir))


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible splits and training behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested training device."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    return device


def build_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> TrainingConfig:
    """Build a TrainingConfig from YAML plus explicit overrides."""

    values = asdict(TrainingConfig())
    if config_path:
        values.update(load_config(config_path))

    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})

    return TrainingConfig(**values)


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Create the configured learning-rate scheduler."""

    scheduler_name = config.scheduler.lower()
    if scheduler_name in {"none", "off"}:
        return None
    if scheduler_name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    raise ValueError("scheduler must be one of: reduce_on_plateau, cosine, none")


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Read the first optimizer parameter group's learning rate."""

    return float(optimizer.param_groups[0]["lr"])


def run_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = False,
    description: str = "train",
) -> EpochMetrics:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=is_training and use_amp and device.type == "cuda")

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        progress = tqdm(dataloader, desc=description, leave=False)
        for batch in progress:
            images, labels, _ = move_batch_to_device(batch, device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=is_training and use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            correct, count = accuracy_from_logits(logits.detach(), labels)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += count

            progress.set_postfix(
                loss=total_loss / max(total_samples, 1),
                accuracy=total_correct / max(total_samples, 1),
            )

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
    )


def predict(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Collect labels and model predictions for a DataLoader."""

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="predict", leave=False):
            images, labels, _ = move_batch_to_device(batch, device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred


def save_checkpoint(
    output_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    config: TrainingConfig,
    class_to_idx: dict[str, int],
) -> None:
    """Save a checkpoint with model and training state."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": epoch_metrics_to_dict(train_metrics),
        "validation_metrics": epoch_metrics_to_dict(validation_metrics),
        "config": asdict(config),
        "class_to_idx": class_to_idx,
    }
    torch.save(checkpoint, output_path)


def load_model_weights(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    """Load model weights from a saved checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def run_training(config_path: str | Path | None = None, **overrides) -> dict[str, Any]:
    """Run the full brain tumor classification training pipeline."""

    config = build_config(config_path=config_path, overrides=overrides)
    set_seed(config.seed)

    run_name = config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.output_dir) / run_name
    checkpoint_dir = Path(config.checkpoint_dir) / run_name
    log_dir = output_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_json(asdict(config), output_dir / "training_config.json")

    device = resolve_device(config.device)
    logger.info("Using device: %s", device)

    loaders = create_brain_tumor_dataloaders(
        dataset_root=config.dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        val_size=config.val_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    logger.info("Class mapping: %s", loaders.class_to_idx)
    logger.info("Split sizes: %s", loaders.split_sizes)
    save_json(loaders.class_to_idx, output_dir / "class_to_idx.json")
    save_json(loaders.split_sizes, output_dir / "split_sizes.json")

    model = create_model(
        architecture=config.architecture,
        num_classes=len(loaders.class_names),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
    ).to(device)
    parameter_counts = count_parameters(model)
    logger.info("Model: %s | parameters: %s", config.architecture, parameter_counts)
    save_json(parameter_counts, output_dir / "parameter_counts.json")

    if config.dry_run:
        images, labels = next(iter(loaders.train))
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        logger.info("Dry run image batch: %s", tuple(images.shape))
        logger.info("Dry run label batch: %s", tuple(labels.shape))
        logger.info("Dry run logits: %s", tuple(logits.shape))
        return {
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "split_sizes": loaders.split_sizes,
            "class_to_idx": loaders.class_to_idx,
        }

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = create_scheduler(optimizer, config)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta, mode="min")
    writer = create_summary_writer(log_dir, enabled=config.tensorboard)
    writer.add_text("config", str(asdict(config)))

    history: list[dict[str, Any]] = []
    best_checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"

    try:
        for epoch in range(1, config.epochs + 1):
            logger.info("Epoch %s/%s", epoch, config.epochs)

            train_metrics = run_one_epoch(
                model=model,
                dataloader=loaders.train,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                use_amp=config.use_amp,
                description=f"train {epoch}/{config.epochs}",
            )
            validation_metrics = run_one_epoch(
                model=model,
                dataloader=loaders.validation,
                criterion=criterion,
                device=device,
                optimizer=None,
                use_amp=False,
                description=f"val {epoch}/{config.epochs}",
            )

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(validation_metrics.loss)
                else:
                    scheduler.step()

            learning_rate = current_learning_rate(optimizer)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
                "learning_rate": learning_rate,
            }
            history.append(row)
            save_history(history, output_dir / "history.csv")

            writer.add_scalar("loss/train", train_metrics.loss, epoch)
            writer.add_scalar("loss/validation", validation_metrics.loss, epoch)
            writer.add_scalar("accuracy/train", train_metrics.accuracy, epoch)
            writer.add_scalar("accuracy/validation", validation_metrics.accuracy, epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)

            save_checkpoint(
                output_path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                config=config,
                class_to_idx=loaders.class_to_idx,
            )

            improved = early_stopping.step(validation_metrics.loss)
            if improved:
                save_checkpoint(
                    output_path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    config=config,
                    class_to_idx=loaders.class_to_idx,
                )
                logger.info("Saved new best checkpoint: %s", best_checkpoint_path)

            logger.info(
                "epoch=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.6g",
                epoch,
                train_metrics.loss,
                train_metrics.accuracy,
                validation_metrics.loss,
                validation_metrics.accuracy,
                learning_rate,
            )

            if early_stopping.should_stop:
                logger.info("Early stopping triggered after %s epochs without improvement.", config.patience)
                break
    finally:
        writer.close()

    if best_checkpoint_path.exists():
        load_model_weights(model, best_checkpoint_path, device=device)

    y_true, y_pred = predict(model, loaders.test, device=device)
    report = save_classification_outputs(
        y_true=y_true,
        y_pred=y_pred,
        class_names=loaders.class_names,
        output_dir=output_dir,
        prefix="test",
    )
    save_json({"history": history, "test_report": report}, output_dir / "training_summary.json")

    logger.info("Training complete. Outputs saved to: %s", output_dir)
    logger.info("Checkpoints saved to: %s", checkpoint_dir)

    return {
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "history": history,
        "test_report": report,
        "class_to_idx": loaders.class_to_idx,
        "split_sizes": loaders.split_sizes,
    }
