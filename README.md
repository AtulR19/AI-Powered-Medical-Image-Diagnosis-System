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

## Training Pipeline

Run a quick offline smoke test first:

```powershell
& .\.venv\Scripts\python.exe train.py --dry-run --no-pretrained --batch-size 2
```

Train a ResNet50 transfer-learning baseline:

```powershell
& .\.venv\Scripts\python.exe train.py --config configs\brain_tumor_resnet50.yaml
```

Train an EfficientNet baseline:

```powershell
& .\.venv\Scripts\python.exe train.py --config configs\brain_tumor_efficientnet_b0.yaml
```

Training outputs are saved under `outputs/training/<run_name>/` and checkpoints under `models/checkpoints/<run_name>/`. The pipeline saves `best.pt`, `latest.pt`, `history.csv`, TensorBoard logs, test confusion matrix, and classification reports.

View TensorBoard logs:

```powershell
& .\.venv\Scripts\tensorboard.exe --logdir outputs\training
```

## Inference Pipeline

Run prediction from the terminal:

```powershell
& .\.venv\Scripts\python.exe predict.py --checkpoint models\checkpoints\resnet50_run1\best.pt --image "data\archive (1)\Testing\glioma\Te-gl_1.jpg"
```

The prediction output includes `predicted_class`, `predicted_index`, `confidence`, and class-wise `probabilities`.

Serve the upload API:

```powershell
$env:MODEL_CHECKPOINT="models\checkpoints\resnet50_run1\best.pt"
& .\.venv\Scripts\uvicorn.exe src.api.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` and use the `/predict` endpoint to upload an MRI image.

Run the Streamlit upload dashboard:

```powershell
& .\.venv\Scripts\streamlit.exe run dashboard\app.py
```

## Grad-CAM Explainability

Generate heatmap, overlay, confidence chart, and summary visualization:

```powershell
& .\.venv\Scripts\python.exe gradcam.py --checkpoint models\checkpoints\20260524-103930\best.pt --image "data\archive (1)\Testing\glioma\Te-gl_1.jpg" --output-dir outputs\gradcam\glioma_example
```

Saved files include `*_gradcam_heatmap.png`, `*_gradcam_overlay.png`, `*_confidence.png`, `*_gradcam_summary.png`, and `*_gradcam_metadata.json`.

## Notes

- Keep patient data and large imaging files out of version control.
- Store raw images in `data/raw/` and generated artifacts in `outputs/`.
- Put reusable pipeline code under `src/` and exploratory work under `notebooks/`.
