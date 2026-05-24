"""Streamlit dashboard for brain tumor MRI diagnosis."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

HISTORY_PATH = PROJECT_ROOT / "outputs" / "dashboard" / "prediction_history.jsonl"
GRADCAM_ROOT = PROJECT_ROOT / "outputs" / "gradcam" / "dashboard"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page: #080c12;
            --surface: #0f1724;
            --surface-strong: #132033;
            --surface-muted: #0b111c;
            --border: #25364f;
            --ink: #edf5ff;
            --muted: #9fb0c7;
            --accent: #3b82f6;
            --accent-strong: #60a5fa;
            --accent-soft: rgba(59, 130, 246, 0.15);
            --ok: #38d39f;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 8%, rgba(59, 130, 246, 0.18), transparent 28rem),
                radial-gradient(circle at 82% 18%, rgba(14, 165, 233, 0.10), transparent 24rem),
                linear-gradient(135deg, #05070b 0%, #08111d 48%, #0e1624 100%);
            color: var(--ink);
        }
        .main .block-container {
            padding-top: 1.25rem;
            max-width: 1320px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #05070b 0%, #0d1625 100%);
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stMarkdown p,
        label, [data-testid="stWidgetLabel"] {
            color: var(--ink);
        }
        .fade-in {
            animation: fadeInUp 640ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .welcome-shell {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(59, 130, 246, 0.03) 34%),
                linear-gradient(160deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            padding: 38px 42px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
        }
        .welcome-shell::after {
            content: "";
            position: absolute;
            inset: auto -12% -26% auto;
            width: 360px;
            height: 360px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.22), transparent 68%);
            pointer-events: none;
        }
        .hero-kicker {
            color: #8dbbff;
            font-size: 0.82rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 16px;
        }
        .hero-title {
            color: var(--ink);
            font-size: clamp(2.35rem, 5vw, 4.8rem);
            line-height: 1.02;
            font-weight: 800;
            max-width: 820px;
            margin: 0;
        }
        .hero-copy {
            color: #cad3df;
            max-width: 650px;
            font-size: 1.08rem;
            line-height: 1.65;
            margin-top: 18px;
        }
        .heartbeat-line {
            height: 56px;
            margin-top: 34px;
            border-radius: 10px;
            background:
                linear-gradient(90deg, transparent 0 7%, rgba(96, 165, 250, 0.48) 7% 8%, transparent 8% 16%),
                linear-gradient(135deg, transparent 0 42%, rgba(96, 165, 250, 0.86) 42% 44%, transparent 44% 47%, rgba(96, 165, 250, 0.86) 47% 49%, transparent 49% 100%),
                linear-gradient(90deg, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0.02));
            border: 1px solid rgba(96, 165, 250, 0.18);
            position: relative;
        }
        .heartbeat-line::before {
            content: "";
            position: absolute;
            left: 0;
            top: 50%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.82), transparent);
            animation: pulseSweep 2.8s ease-in-out infinite;
        }
        @keyframes pulseSweep {
            0% { transform: translateX(-45%) scaleX(0.35); opacity: 0; }
            35% { opacity: 1; }
            100% { transform: translateX(45%) scaleX(1); opacity: 0; }
        }
        .feature-card,
        .mini-card {
            background: rgba(15, 23, 36, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.22);
            transition: transform 220ms ease, border-color 220ms ease, background 220ms ease;
        }
        .feature-card:hover,
        .mini-card:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.46);
            background: rgba(19, 32, 51, 0.94);
        }
        .feature-card strong,
        .mini-card strong {
            color: var(--ink);
            display: block;
            font-size: 1.02rem;
            margin-bottom: 7px;
        }
        .feature-card span,
        .mini-card span {
            color: var(--muted);
            line-height: 1.5;
            font-size: 0.94rem;
        }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.22);
            transition: transform 180ms ease, border-color 180ms ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-1px);
            border-color: rgba(96, 165, 250, 0.46);
        }
        div[data-testid="stMetric"],
        div[data-testid="stMetric"] * {
            color: var(--ink) !important;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-weight: 650 !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricValue"] div {
            color: var(--ink) !important;
            font-weight: 760 !important;
        }
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {
            color: var(--ok) !important;
        }
        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.22);
        }
        .panel,
        .panel * {
            color: var(--ink) !important;
        }
        .subtle {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .status-pill {
            display: inline-block;
            border: 1px solid rgba(72, 213, 151, 0.38);
            background: rgba(72, 213, 151, 0.12);
            color: #8cf0c0;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.86rem;
            font-weight: 650;
        }
        .result-title {
            color: var(--ink);
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .danger-note {
            border-left: 4px solid var(--accent);
            background: rgba(59, 130, 246, 0.10);
            padding: 10px 12px;
            border-radius: 6px;
            color: #dceaff;
        }
        .stButton > button {
            border-radius: 10px;
            min-height: 48px;
            font-weight: 760;
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 28px rgba(59, 130, 246, 0.24);
            filter: brightness(1.04);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 18px;
            margin: 22px 0 8px;
        }
        .status-card {
            min-width: 0;
            background: rgba(15, 23, 36, 0.92);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.20);
        }
        .status-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .status-value {
            color: var(--ink);
            font-size: clamp(1rem, 1.7vw, 1.35rem);
            font-weight: 760;
            line-height: 1.25;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .status-caption {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 8px;
        }
        @media (max-width: 980px) {
            .status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .welcome-shell {
                padding: 28px;
            }
        }
        @media (max-width: 640px) {
            .status-grid {
                grid-template-columns: 1fr;
            }
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }
        .stAlert {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_checkpoints() -> list[Path]:
    checkpoint_root = PROJECT_ROOT / "models" / "checkpoints"
    if not checkpoint_root.exists():
        return []
    checkpoints = sorted(checkpoint_root.rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    checkpoints.extend(
        path for path in sorted(checkpoint_root.rglob("latest.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        if path not in checkpoints
    )
    return checkpoints


def to_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_checkpoint_path(selection: str, manual_path: str) -> Path | None:
    candidate = manual_path.strip() or selection
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint_path: str, device: str):
    from src.inference import BrainTumorPredictor

    return BrainTumorPredictor(checkpoint_path=checkpoint_path, device=device)


def show_missing_dependency_error(error: ModuleNotFoundError) -> None:
    st.error(f"Missing Python package: {error.name}")
    st.code(r"& .\.venv\Scripts\python.exe -m streamlit run dashboard\app.py", language="powershell")
    st.caption("Stop the current Streamlit server first if it was launched from Anaconda or another Python environment.")


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()

    rows = []
    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    dataframe = pd.DataFrame(rows)
    if "timestamp" in dataframe.columns:
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    return dataframe


def append_history(record: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def render_probability_table(probabilities: dict[str, float]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "class": list(probabilities.keys()),
            "probability": list(probabilities.values()),
        }
    )
    return table.sort_values("probability", ascending=False).reset_index(drop=True)


def render_sidebar() -> tuple[Path | None, str]:
    st.sidebar.title("Diagnosis Console")
    page = st.sidebar.radio("Page", ["Welcome", "Diagnosis", "Analytics", "History"], label_visibility="collapsed")

    checkpoints = find_checkpoints()
    checkpoint_labels = [to_display_path(path) for path in checkpoints]
    default_selection = checkpoint_labels[0] if checkpoint_labels else ""
    selected_checkpoint = st.sidebar.selectbox(
        "Checkpoint",
        options=checkpoint_labels,
        index=0 if checkpoint_labels else None,
        placeholder="No checkpoints found",
    )
    manual_checkpoint = st.sidebar.text_input("Custom checkpoint", value="" if checkpoint_labels else default_selection)
    device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    checkpoint_path = resolve_checkpoint_path(selected_checkpoint or "", manual_checkpoint)
    if checkpoint_path and checkpoint_path.exists():
        st.sidebar.markdown(f'<span class="status-pill">Model ready</span>', unsafe_allow_html=True)
        st.sidebar.caption(to_display_path(checkpoint_path))
    else:
        st.sidebar.warning("Checkpoint unavailable")

    st.session_state["device"] = device
    return checkpoint_path, page


def get_runtime_status() -> dict[str, str]:
    try:
        import torch

        cuda_ready = torch.cuda.is_available()
        return {
            "device": torch.cuda.get_device_name(0) if cuda_ready else "CPU",
            "torch": torch.__version__,
            "cuda": "Ready" if cuda_ready else "Unavailable",
        }
    except ModuleNotFoundError:
        return {"device": "Unavailable", "torch": "Missing", "cuda": "Unavailable"}


def render_welcome_page(checkpoint_path: Path | None) -> None:
    status = get_runtime_status()
    model_status = "Ready" if checkpoint_path and checkpoint_path.exists() else "Select checkpoint"
    checkpoint_label = to_display_path(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else "No checkpoint selected"

    st.markdown(
        """
        <section class="welcome-shell fade-in">
            <div class="hero-kicker">Brain Tumor Diagnosis System</div>
            <h1 class="hero-title">Your MRI. Our signal.</h1>
            <p class="hero-copy">
                A focused diagnostic workspace for MRI classification, confidence review,
                and visual model evidence.
            </p>
            <div class="heartbeat-line"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-grid fade-in">
            <div class="status-card">
                <div class="status-label">Model</div>
                <div class="status-value">{model_status}</div>
                <div class="status-caption">{checkpoint_label}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Runtime</div>
                <div class="status-value">{status["device"]}</div>
                <div class="status-caption">Selected from dashboard sidebar</div>
            </div>
            <div class="status-card">
                <div class="status-label">CUDA</div>
                <div class="status-value">{status["cuda"]}</div>
                <div class="status-caption">GPU acceleration status</div>
            </div>
            <div class="status-card">
                <div class="status-label">PyTorch</div>
                <div class="status-value">{status["torch"]}</div>
                <div class="status-caption">Active environment build</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    card_1, card_2, card_3 = st.columns(3)
    card_1.markdown(
        """
        <div class="feature-card fade-in">
            <strong>Diagnosis Workspace</strong>
            <span>Upload MRI images and review the predicted tumor class with calibrated probabilities.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_2.markdown(
        """
        <div class="feature-card fade-in">
            <strong>Grad-CAM Evidence</strong>
            <span>Inspect heatmaps and overlays that highlight the regions driving the model response.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_3.markdown(
        """
        <div class="feature-card fade-in">
            <strong>Longitudinal Review</strong>
            <span>Track prediction history, confidence trends, and class distribution from dashboard usage.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        '<div class="danger-note">Decision support only. Results require qualified medical review.</div>',
        unsafe_allow_html=True,
    )


def render_diagnosis_page(checkpoint_path: Path | None) -> None:
    st.markdown(
        """
        <div class="fade-in">
            <div class="hero-kicker">Diagnosis</div>
            <h1 style="margin: 0 0 8px 0; color: var(--ink);">Brain Tumor Diagnosis</h1>
            <p class="subtle">Upload an MRI scan, run classification, and review model evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if checkpoint_path is None or not checkpoint_path.exists():
        st.error("Select a valid checkpoint before running prediction.")
        return

    uploaded_file = st.file_uploader(
        "MRI image",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.markdown('<div class="panel subtle">Awaiting MRI image upload.</div>', unsafe_allow_html=True)
        return

    image_bytes = uploaded_file.getvalue()
    upload_signature = sha256(image_bytes).hexdigest()
    if st.session_state.get("last_upload_signature") != upload_signature:
        st.session_state.pop("last_result", None)
        st.session_state["last_upload_signature"] = upload_signature

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output_dir = GRADCAM_ROOT / run_id
    output_stem = Path(uploaded_file.name).stem.replace(" ", "_")

    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        st.markdown('<div class="result-title">MRI Image</div>', unsafe_allow_html=True)
        st.image(Image.open(uploaded_file).convert("RGB"), caption=uploaded_file.name, use_container_width=True)

    with right:
        st.markdown('<div class="result-title">Model Output</div>', unsafe_allow_html=True)
        if st.button("Run Diagnosis", type="primary", use_container_width=True):
            with st.spinner("Running prediction and Grad-CAM..."):
                try:
                    from src.explainability import explain_with_predictor

                    predictor = load_predictor(str(checkpoint_path), st.session_state.get("device", "auto"))
                    result = explain_with_predictor(
                        predictor=predictor,
                        image_path=image_bytes,
                        output_dir=output_dir,
                        output_stem=output_stem,
                    )
                except ModuleNotFoundError as exc:
                    show_missing_dependency_error(exc)
                    return

            append_history(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "filename": uploaded_file.name,
                    "checkpoint": to_display_path(checkpoint_path),
                    "predicted_class": result.predicted_class,
                    "target_class": result.target_class,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "heatmap_path": result.heatmap_path,
                    "overlay_path": result.overlay_path,
                    "confidence_path": result.confidence_path,
                    "summary_path": result.summary_path,
                }
            )
            st.session_state["last_result"] = result.to_dict()

        result_data = st.session_state.get("last_result")
        if result_data:
            col_a, col_b = st.columns(2)
            col_a.metric("Prediction", result_data["predicted_class"])
            col_b.metric("Confidence", f"{result_data['confidence']:.2%}")

            probability_table = render_probability_table(result_data["probabilities"])
            st.bar_chart(probability_table.set_index("class"))
            st.dataframe(
                probability_table.assign(probability=lambda frame: frame["probability"].map(lambda value: f"{value:.2%}")),
                use_container_width=True,
                hide_index=True,
            )

    result_data = st.session_state.get("last_result")
    if result_data:
        st.markdown('<div class="hero-kicker" style="margin-top: 26px;">Explainability</div>', unsafe_allow_html=True)
        st.subheader("Grad-CAM")
        image_col_1, image_col_2, image_col_3 = st.columns(3)
        image_col_1.image(result_data["heatmap_path"], caption="Heatmap", use_container_width=True)
        image_col_2.image(result_data["overlay_path"], caption="Overlay", use_container_width=True)
        image_col_3.image(result_data["confidence_path"], caption="Confidence", use_container_width=True)
        st.image(result_data["summary_path"], caption="Summary", use_container_width=True)
        st.markdown(
            '<div class="danger-note">Prediction support only. Clinical decisions require qualified medical review.</div>',
            unsafe_allow_html=True,
        )


def render_analytics_page() -> None:
    st.header("Analytics")
    history = load_history()

    if history.empty:
        st.info("No prediction history yet.")
        return

    total_predictions = len(history)
    average_confidence = float(history["confidence"].mean())
    latest_prediction = history.sort_values("timestamp").iloc[-1]
    most_common_class = history["predicted_class"].mode().iloc[0]

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Predictions", f"{total_predictions:,}")
    metric_2.metric("Average confidence", f"{average_confidence:.2%}")
    metric_3.metric("Most frequent class", str(most_common_class))
    metric_4.metric("Latest class", str(latest_prediction["predicted_class"]))

    class_counts = history["predicted_class"].value_counts().rename_axis("class").reset_index(name="count")
    st.subheader("Class Distribution")
    st.bar_chart(class_counts.set_index("class"))

    confidence_timeline = history[["timestamp", "confidence"]].dropna().sort_values("timestamp")
    if not confidence_timeline.empty:
        st.subheader("Confidence Over Time")
        st.line_chart(confidence_timeline.set_index("timestamp"))

    probability_rows = []
    for _, row in history.iterrows():
        probabilities = row.get("probabilities", {})
        if isinstance(probabilities, str):
            probabilities = json.loads(probabilities)
        for class_name, probability in probabilities.items():
            probability_rows.append({"class": class_name, "probability": probability})

    if probability_rows:
        probability_frame = pd.DataFrame(probability_rows)
        st.subheader("Mean Class Probability")
        st.bar_chart(probability_frame.groupby("class")["probability"].mean())


def render_history_page() -> None:
    st.header("Prediction History")
    history = load_history()

    if history.empty:
        st.info("No saved predictions.")
        return

    display_columns = [
        "timestamp",
        "filename",
        "predicted_class",
        "confidence",
        "checkpoint",
        "overlay_path",
        "summary_path",
    ]
    visible = history[[column for column in display_columns if column in history.columns]].copy()
    if "confidence" in visible.columns:
        visible["confidence"] = visible["confidence"].map(lambda value: f"{float(value):.2%}")

    st.dataframe(visible.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

    latest = history.sort_values("timestamp").iloc[-1]
    preview_path = latest.get("summary_path")
    if isinstance(preview_path, str) and Path(preview_path).exists():
        st.subheader("Latest Summary")
        st.image(preview_path, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Brain Tumor Diagnosis",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    checkpoint_path, page = render_sidebar()

    if page == "Welcome":
        render_welcome_page(checkpoint_path)
    elif page == "Diagnosis":
        render_diagnosis_page(checkpoint_path)
    elif page == "Analytics":
        render_analytics_page()
    else:
        render_history_page()


if __name__ == "__main__":
    main()
