"""Dashboard entry point for model monitoring and result review."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Medical Image Diagnosis", layout="wide")
    st.title("Medical Image Diagnosis")
    st.write("Use this dashboard to review predictions, explanations, and model metrics.")


if __name__ == "__main__":
    main()
