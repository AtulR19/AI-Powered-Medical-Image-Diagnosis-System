"""FastAPI application for serving diagnosis predictions."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Medical Image Diagnosis API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
