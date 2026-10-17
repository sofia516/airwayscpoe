"""Streamlit interface for the AirwayScope academic research demo."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from airway.annotate import annotate_image
from airway.calibration import calibrate_scale
from airway.image_io import load_image
from airway.landmarks import detect_landmarks
from airway.measurements import measure_airway_features
from airway.quality import validate_quality
from airway.storage import export_csv, load_records, save_record


st.set_page_config(page_title="AirwayScope", page_icon="A", layout="wide")

st.markdown(
    """
    <style>
      :root { --ink:#172a3a; --muted:#6b7c8b; --teal:#087f8c; --line:#e1e9ed; --paper:#f5f8f9; }
      html, body, [class*="css"] { font-family:'Segoe UI',Arial,sans-serif; }
      .stApp { background:var(--paper); }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { max-width:1240px; padding-top:1.35rem; padding-bottom:3rem; }
      h1,h2,h3 { font-family:'Segoe UI',Arial,sans-serif; color:var(--ink); letter-spacing:-.035em; }
      .brand-row { display:flex; align-items:center; gap:12px; margin:0 0 1.5rem; }
      .brand-mark { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; color:white; background:linear-gradient(145deg,#16a2a2,#087f8c); font-size:22px; font-weight:700; box-shadow:0 7px 18px #087f8c2b; }
      .brand-name { font:800 21px 'Segoe UI',Arial,sans-serif; letter-spacing:-.04em; color:var(--ink); }
      .brand-sub { color:var(--muted); font-size:12px; margin-top:1px; }
      .hero { display:flex; justify-content:space-between; align-items:flex-end; gap:20px; padding:1.55rem 1.8rem; margin-bottom:1.2rem; border:1px solid #dce8eb; border-radius:22px; background:linear-gradient(115deg,#eaf7f6 0%,#f7fbfb 54%,#edf3f8 100%); }
      .eyebrow { color:var(--teal); font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:7px; }
      .hero h1 { font-size:clamp(28px,3.5vw,40px); line-height:1.12; margin:0; }
      .hero p { max-width:610px; color:#526777; margin:.55rem 0 0; font-size:14px; line-height:1.55; }
      .demo-pill { white-space:nowrap; border:1px solid #c7dfdf; border-radius:999px; color:#286568; background:#ffffffb5; padding:8px 12px; font-size:11px; font-weight:700; }
      .section-kicker { color:#718390; font-size:11px; text-transform:uppercase; letter-spacing:.1em; font-weight:700; margin:8px 0 10px; }
      .panel { background:white; border:1px solid var(--line); border-radius:17px; padding:18px 20px; }
      .helper { color:var(--muted); font-size:12px; line-height:1.5; }
      [data-testid="stFileUploaderDropzone"] { background:#f8fbfc; border:1px dashed #9eb9c0; border-radius:14px; }
      [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--teal); background:#f1fbfa; }
      [data-testid="stFileUploaderDropzoneInstructions"] { color:#4f6874; }
      .metric-card { height:100%; min-height:116px; padding:16px; background:white; border:1px solid var(--line); border-radius:15px; }
      .metric-label { color:#728390; font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
      .metric-value { color:var(--ink); font:700 25px 'Segoe UI',Arial,sans-serif; letter-spacing:-.04em; margin-top:9px; }
      .metric-unit { color:#7a8a94; font:500 12px 'Segoe UI',Arial,sans-serif; letter-spacing:0; }
      .privacy-note { border-left:3px solid #50a4a2; color:#557078; padding:7px 11px; font-size:12px; line-height:1.45; background:#f4faf9; border-radius:0 8px 8px 0; }
      div[data-testid="stTabs"] button { font-weight:700; }
      div.stButton > button[kind="primary"] { border-radius:10px; background:#087f8c; border-color:#087f8c; }
      div.stButton > button { border-radius:10px; }
      [data-testid="stAlert"] { border-radius:12px; }
      footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, unit: str) -> None:
    """Render a compact measurement card."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value} <span class="metric-unit">{unit}</span></div></div>',
        unsafe_allow_html=True,
    )


def show_measurements(measurements: dict[str, float]) -> None:
    """Render the five measured values in a responsive card grid."""
    items = [
        ("Mouth opening", "mouth_opening_mm", "mm", 1),
        ("Mouth width", "mouth_width_mm", "mm", 1),
        ("Jaw width", "jaw_width_mm", "mm", 1),
        ("Face height proxy", "face_height_mm", "mm", 1),
        ("Opening / height", "mouth_opening_to_face_height_ratio", "ratio", 3),
    ]
    rows = [st.columns(3), st.columns(2)]
    for index, (label, key, unit, digits) in enumerate(items):
        row = rows[0] if index < 3 else rows[1]
        col = row[index] if index < 3 else row[index - 3]
        value = f"{measurements[key]:.{digits}f}"
        with col:
            metric_card(label, value, unit)


def run_analysis(image: np.ndarray, calibration: str) -> None:
    """Detect, calibrate, measure and display the selected image."""
    try:
        landmarks = detect_landmarks(image)
        if landmarks is None:
            st.error("No face detected. Try a clear, front-facing image with the full face in frame.")
            return

        if calibration == "Known reference object":
            points = (
                (st.session_state.ref_x1, st.session_state.ref_y1),
                (st.session_state.ref_x2, st.session_state.ref_y2),
            )
            scale, method = calibrate_scale(
                image, "reference", points, st.session_state.reference_mm, landmarks
            )
        else:
            scale, method = calibrate_scale(image, "iris", landmarks=landmarks)

        measurements = measure_airway_features(landmarks, scale)
        warnings = validate_quality(image, landmarks)
        annotated = annotate_image(image, landmarks, measurements)

        st.markdown('<div class="section-kicker">Analysis result</div>', unsafe_allow_html=True)
        show_measurements(measurements)
        st.caption(f"Scale: {scale:.5f} mm/px · Calibration: {method}")
        if method.startswith("iris"):
            st.info("Iris scaling is an approximate estimate (about ±10%). A reference object can improve scale calibration.")

        photo_col, overlay_col = st.columns(2, gap="large")
        with photo_col:
            st.markdown('<div class="section-kicker">Original photo</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
        with overlay_col:
            st.markdown('<div class="section-kicker">Landmark overlay</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

        if warnings:
            with st.container(border=True):
                st.markdown("**Image quality checks**")
                for warning in warnings:
                    st.warning(warning)
        else:
            st.success("The image passed the demo’s basic quality checks.")

        if st.button("Save this measurement", type="primary"):
            case_id = save_record(None, measurements, method, warnings)
            st.success(f"Saved anonymous case {case_id}. The photo was not saved.")
    except ValueError as exc:
        st.error(str(exc))
    except RuntimeError as exc:
        st.error(str(exc))


def render_analysis_tab() -> None:
    """Render the photo upload and analysis workflow."""
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="section-kicker">01 · Add a photo</div>', unsafe_allow_html=True)
        source = st.radio("Photo source", ["Upload photo", "Use camera"], horizontal=True, label_visibility="collapsed")
        uploaded = (
            st.file_uploader("Choose a face photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            if source == "Upload photo"
            else st.camera_input("Take a face photo")
        )
        st.markdown(
            '<div class="helper">Use a clear, front-facing image with the full face visible. Minimum image size: 480 px on the short side.</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-kicker">02 · Choose scale</div>', unsafe_allow_html=True)
        calibration = st.selectbox(
            "Calibration method",
            ["Iris estimate (quick)", "Known reference object"],
            label_visibility="collapsed",
        )
        if calibration == "Known reference object":
            st.number_input("Known object length (mm)", min_value=0.1, value=25.0, step=1.0, key="reference_mm")
            st.markdown(
                '<div class="helper">After the photo loads, enter the pixel coordinates of the object’s two endpoints.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="helper">Uses an average iris diameter assumption. Approximate only; a known object is preferred when available.</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="privacy-note">Your photo is processed in memory. It is not stored with saved measurements.</div>', unsafe_allow_html=True)
    if uploaded is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Add a photo to see the measurement summary and landmark overlay here.")
        return

    try:
        image = load_image(uploaded)
        if calibration == "Known reference object":
            height, width = image.shape[:2]
            with st.container(border=True):
                st.markdown("**Reference object coordinates**")
                st.caption(f"Photo dimensions: {width} × {height} px. Choose two points along an object of known length.")
                coords = st.columns(4)
                fields = [
                    ("Point 1 · x", "ref_x1", 100.0, width),
                    ("Point 1 · y", "ref_y1", 100.0, height),
                    ("Point 2 · x", "ref_x2", 200.0, width),
                    ("Point 2 · y", "ref_y2", 100.0, height),
                ]
                for col, (label, key, default, max_value) in zip(coords, fields):
                    with col:
                        st.number_input(label, min_value=0.0, max_value=float(max_value), value=min(float(default), float(max_value)), key=key)
        run_analysis(image, "Known reference object" if calibration == "Known reference object" else "Iris estimate")
    except ValueError as exc:
        st.error(str(exc))


def render_records_tab() -> None:
    """Render saved anonymous measurements and export controls."""
    st.markdown('<div class="section-kicker">Saved research records</div>', unsafe_allow_html=True)
    records = load_records()
    if records.empty:
        st.info("No saved measurements yet. Run an analysis and choose “Save this measurement” to add a record.")
        return
    st.dataframe(records, width="stretch", hide_index=True)
    numeric = records.select_dtypes(include="number")
    if not numeric.empty:
        with st.expander("Summary statistics"):
            st.dataframe(numeric.describe().T, width="stretch")
    st.download_button(
        "Download CSV", data=export_csv(), file_name="airwayscope_records.csv", mime="text/csv"
    )


def main() -> None:
    """Render the AirwayScope analysis and records views."""
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">◉</div><div><div class="brand-name">AirwayScope</div>'
        '<div class="brand-sub">Facial measurement research demo</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><div><div class="eyebrow">Photo-based measurement</div><h1>A clearer view of each measurement.</h1>'
        '<p>Add a face photo, choose a scale method, and review the landmark measurements with image quality notes.</p></div>'
        '<div class="demo-pill">RESEARCH DEMO · NOT FOR CLINICAL USE</div></div>',
        unsafe_allow_html=True,
    )
    analysis_tab, records_tab = st.tabs(["Measure a photo", "Saved records"])
    with analysis_tab:
        render_analysis_tab()
    with records_tab:
        render_records_tab()
    st.markdown(
        '<div class="helper" style="text-align:center;margin-top:30px">Experimental estimates only. Not clinically validated and not for diagnosis or treatment.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
