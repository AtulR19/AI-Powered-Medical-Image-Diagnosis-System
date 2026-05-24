"""FastAPI application for serving diagnosis predictions."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference import BrainTumorPredictor

app = FastAPI(title="Medical Image Diagnosis API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


def get_checkpoint_path() -> Path:
    """Read model checkpoint path from environment."""

    checkpoint = os.getenv("MODEL_CHECKPOINT") or os.getenv("BRAIN_TUMOR_MODEL_PATH")
    if not checkpoint:
        raise HTTPException(
            status_code=503,
            detail="Set MODEL_CHECKPOINT to a trained best.pt or latest.pt file before prediction.",
        )
    return Path(checkpoint)


@lru_cache(maxsize=1)
def get_predictor(checkpoint_path: str) -> BrainTumorPredictor:
    """Load and cache the model predictor."""

    try:
        return BrainTumorPredictor(checkpoint_path=checkpoint_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
async def predict_brain_tumor(file: UploadFile = File(...)) -> dict:
    """Predict tumor class from an uploaded MRI image."""

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    predictor = get_predictor(str(get_checkpoint_path()))

    try:
        result = predictor.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = result.to_dict()
    response["filename"] = file.filename
    return response
