# Medical Image Diagnosis System

A modular PyTorch project scaffold for building medical image diagnosis workflows, including preprocessing, model training, inference, explainability, API serving, and dashboard development.

## Project Structure

```text
medical-image-diagnosis-system/
|-- data/
|   |-- raw/            # Original datasets
|   |-- interim/        # Intermediate preprocessing outputs
|   |-- processed/      # Model-ready datasets
|   `-- external/       # External reference data
|-- notebooks/          # Research, EDA, and experiment notebooks
|-- models/
|   |-- checkpoints/    # Training checkpoints
|   `-- exports/        # Exported production models
|-- outputs/
|   |-- logs/           # Training and inference logs
|   |-- reports/        # Evaluation reports
|   `-- figures/        # Plots and explainability images
|-- dashboard/          # Streamlit or dashboard UI code
|-- src/
|   |-- preprocessing/  # Dataset loading and transforms
|   |-- training/       # Training loops and metrics
|   |-- inference/      # Prediction utilities
|   |-- explainability/ # Grad-CAM and attribution methods
|   |-- utils/          # Config, logging, and shared helpers
|   `-- api/            # FastAPI application
|-- requirements.txt
|-- README.md
`-- train.py
```

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

## Brain Tumor MRI DataLoader Pipeline

```python
from pathlib import Path

from src.preprocessing import create_brain_tumor_dataloaders

loaders = create_brain_tumor_dataloaders(
    dataset_root=Path("data/archive (1)"),
    image_size=224,
    batch_size=32,
    val_size=0.2,
    num_workers=0,
    seed=42,
)

print(loaders.class_to_idx)
print(loaders.split_sizes)

images, labels = next(iter(loaders.train))
print(images.shape)  # torch.Size([32, 3, 224, 224])
print(labels.shape)  # torch.Size([32])
```

The pipeline uses the provided `Testing` folder as the final holdout set and creates a stratified validation split from `Training`. Training images use Albumentations resizing, mild augmentation, normalization, and tensor conversion. Validation and test images use deterministic resizing, normalization, and tensor conversion.

## Notes

- Keep patient data and large imaging files out of version control.
- Store raw images in `data/raw/` and generated artifacts in `outputs/`.
- Put reusable pipeline code under `src/` and exploratory work under `notebooks/`.
