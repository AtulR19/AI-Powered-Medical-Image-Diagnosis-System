"""Dashboard entry point for model monitoring and result review."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.inference import BrainTumorPredictor


@st.cache_resource
def load_predictor(checkpoint_path: str) -> BrainTumorPredictor:
    return BrainTumorPredictor(checkpoint_path=checkpoint_path)


def main() -> None:
    st.set_page_config(page_title="Medical Image Diagnosis", layout="wide")
    st.title("Medical Image Diagnosis")

    checkpoint_path = st.text_input(
        "Model checkpoint",
        value="models/checkpoints/resnet50_run1/best.pt",
    )
    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    if uploaded_file is None:
        return

    st.image(uploaded_file, caption="Uploaded MRI", width=360)

    if not Path(checkpoint_path).exists():
        st.error("Checkpoint not found. Train a model first, then enter the path to best.pt or latest.pt.")
        return

    predictor = load_predictor(checkpoint_path)
    result = predictor.predict(uploaded_file.getvalue())

    st.subheader("Prediction")
    st.metric("Predicted class", result.predicted_class)
    st.metric("Confidence", f"{result.confidence:.2%}")
    st.bar_chart(result.probabilities)


if __name__ == "__main__":
    main()
