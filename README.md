# Medical Image Diagnosis System

A modular PyTorch project scaffold for building medical image diagnosis workflows, including preprocessing, model training, inference, explainability, API serving, and dashboard development.

## Project Structure

```text
medical-image-diagnosis-system/
├── data/
│   ├── raw/            # Original datasets
│   ├── interim/        # Intermediate preprocessing outputs
│   ├── processed/      # Model-ready datasets
│   └── external/       # External reference data
├── notebooks/          # Research, EDA, and experiment notebooks
├── models/
│   ├── checkpoints/    # Training checkpoints
│   └── exports/        # Exported production models
├── outputs/
│   ├── logs/           # Training and inference logs
│   ├── reports/        # Evaluation reports
│   └── figures/        # Plots and explainability images
├── dashboard/          # Streamlit or dashboard UI code
├── src/
│   ├── preprocessing/  # Dataset loading and transforms
│   ├── training/       # Training loops and metrics
│   ├── inference/      # Prediction utilities
│   ├── explainability/ # Grad-CAM and attribution methods
│   ├── utils/          # Config, logging, and shared helpers
│   └── api/            # FastAPI application
├── requirements.txt
├── README.md
└── train.py
```

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

## Notes

- Keep patient data and large imaging files out of version control.
- Store raw images in `data/raw/` and generated artifacts in `outputs/`.
- Put reusable pipeline code under `src/` and exploratory work under `notebooks/`.
