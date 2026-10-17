"""Streamlit interface for the AirwayScope academic research demo."""

from __future__ import annotations

import hashlib

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
      :root { --ink:#172a3a; --muted:#687c89; --teal:#087f8c; --line:#dfe8eb; --paper:#f5f8f9; }
      html, body, [class*="css"] { font-family:'Segoe UI',Arial,sans-serif; }
      .stApp { background:var(--paper); }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { max-width:1180px; padding-top:1.25rem; padding-bottom:3rem; }
      h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
      .brand-row { display:flex; align-items:center; gap:12px; margin:0 0 1.35rem; }
      .brand-mark { display:grid; place-items:center; width:40px; height:40px; border-radius:13px; color:white; background:#087f8c; font-size:20px; font-weight:800; }
      .brand-name { font-size:20px; font-weight:800; letter-spacing:-.04em; color:var(--ink); }
      .brand-sub { color:var(--muted); font-size:12px; margin-top:1px; }
      .hero { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:1.35rem 1.6rem; margin-bottom:1.2rem; border:1px solid #dce8eb; border-radius:20px; background:linear-gradient(115deg,#eaf7f6,#f7fbfb 60%,#edf3f8); }
      .eyebrow { color:var(--teal); font-size:11px; font-weight:700; letter-spacing:.11em; text-transform:uppercase; margin-bottom:6px; }
      .hero h1 { font-size:clamp(25px,3vw,36px); line-height:1.12; margin:0; }
      .hero p { max-width:620px; color:#526777; margin:.45rem 0 0; font-size:14px; line-height:1.55; }
      .demo-pill { white-space:nowrap; border:1px solid #c7dfdf; border-radius:999px; color:#286568; background:#ffffffb5; padding:8px 12px; font-size:10px; font-weight:700; }
      .section-kicker { color:#718390; font-size:11px; text-transform:uppercase; letter-spacing:.1em; font-weight:700; margin:8px 0 10px; }
      .helper { color:var(--muted); font-size:12px; line-height:1.5; }
      [data-testid="stFileUploaderDropzone"] { background:#f8fbfc; border:1px dashed #9eb9c0; border-radius:14px; }
      [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--teal); background:#f1fbfa; }
      .metric-card { min-height:106px; padding:14px; background:white; border:1px solid var(--line); border-radius:14px; }
      .metric-label { color:#728390; font-size:11px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; }
      .metric-value { color:var(--ink); font-size:24px; font-weight:700; letter-spacing:-.04em; margin-top:8px; }
      .metric-unit { color:#7a8a94; font-size:12px; font-weight:500; letter-spacing:0; }
      .privacy-note { border-left:3px solid #50a4a2; color:#557078; padding:8px 11px; font-size:12px; line-height:1.45; background:#f4faf9; border-radius:0 8px 8px 0; }
      div[data-testid="stTabs"] button { font-weight:700; }
      div.stButton > button { border-radius:10px; }
      div.stButton > button[kind="primary"] { border-radius:10px; background:#087f8c; border-color:#087f8c; }
      [data-testid="stAlert"] { border-radius:12px; }
      @media (max-width:700px) { .hero { align-items:flex-start; flex-direction:column; padding:1.1rem; } .demo-pill { white-space:normal; } }
      footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, unit: str) -> None:
    """Render one measurement card."""
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
        with col:
            metric_card(label, f"{measurements[key]:.{digits}f}", unit)


def analyze_image(
    image: np.ndarray,
    calibration: str,
    reference_mm: float,
    points: tuple[tuple[float, float], tuple[float, float]],
) -> dict | None:
    """Calculate measurements after the user submits the analysis step."""
    landmarks = detect_landmarks(image)
    if landmarks is None:
        return None
    if calibration == "Known reference object":
        scale, method = calibrate_scale(image, "reference", points, reference_mm, landmarks)
    else:
        scale, method = calibrate_scale(image, "iris", landmarks=landmarks)
    measurements = measure_airway_features(landmarks, scale)
    return {
        "scale": scale,
        "method": method,
        "measurements": measurements,
        "warnings": validate_quality(image, landmarks),
        "annotated": annotate_image(image, landmarks, measurements),
    }


def render_analysis_result(image: np.ndarray, result: dict) -> None:
    """Render a completed analysis and its save action."""
    measurements = result["measurements"]
    method = result["method"]
    warnings = result["warnings"]

    st.markdown('<div class="section-kicker">04 - Your result</div>', unsafe_allow_html=True)
    show_measurements(measurements)
    st.caption(f"Scale: {result['scale']:.5f} mm/px | Calibration: {method}")
    if method.startswith("iris"):
        st.info("Iris scaling is approximate (about 10%). Use a known reference object when possible.")

    photo_col, overlay_col = st.columns(2, gap="large")
    with photo_col:
        st.markdown("**Original photo**")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    with overlay_col:
        st.markdown("**Landmark overlay**")
        st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), width="stretch")

    if warnings:
        with st.container(border=True):
            st.markdown("**Review image quality**")
            for warning in warnings:
                st.warning(warning)
    else:
        st.success("No basic image quality flags were raised.")

    if st.button("Save this measurement", type="primary", key="save_measurement"):
        case_id = save_record(None, measurements, method, warnings)
        st.session_state.saved_case_id = case_id
        st.success(f"Saved anonymous case {case_id}. The photo was not saved.")
    elif st.session_state.get("saved_case_id"):
        st.success(f"Saved anonymous case {st.session_state.saved_case_id}. The photo was not saved.")


def render_analysis_tab() -> None:
    """Render the photo upload and analysis workflow."""
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown('<div class="section-kicker">01 - Add a photo</div>', unsafe_allow_html=True)
        source = st.radio(
            "Photo source", ["Upload photo", "Use camera"], horizontal=True, label_visibility="collapsed"
        )
        uploaded = (
            st.file_uploader(
                "Choose a face photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
            )
            if source == "Upload photo"
            else st.camera_input("Take a face photo")
        )
        st.markdown(
            '<div class="helper">Full face in frame; front-facing; minimum 480 px on the short side.</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-kicker">02 - Choose scale</div>', unsafe_allow_html=True)
        calibration = st.selectbox(
            "Calibration method",
            ["Iris estimate (quick)", "Known reference object"],
            label_visibility="collapsed",
        )
        if calibration == "Known reference object":
            st.number_input(
                "Known object length (mm)", min_value=0.1, value=25.0, step=1.0, key="reference_mm"
            )
            st.caption("Use an object in the same plane as the face for a better scale.")
        else:
            st.caption("Quick estimate from iris size. Approximate; a known object is preferred.")

    st.markdown(
        '<div class="privacy-note">Your photo is processed in memory. It is not saved with measurements.</div>',
        unsafe_allow_html=True,
    )
    if uploaded is None:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_key", None)
        st.session_state.pop("saved_case_id", None)
        st.info("Add a photo to preview it here. You can review the image before asking for measurements.")
        return

    try:
        image = load_image(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return

    image_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
    reference_mm = float(st.session_state.get("reference_mm", 25.0))
    points = (
        (float(st.session_state.get("ref_x1", 100.0)), float(st.session_state.get("ref_y1", 100.0))),
        (float(st.session_state.get("ref_x2", 200.0)), float(st.session_state.get("ref_y2", 100.0))),
    )

    st.markdown('<div class="section-kicker">03 - Review photo and analyze</div>', unsafe_allow_html=True)
    if calibration == "Known reference object":
        height, width = image.shape[:2]
        preview_col, coords_col = st.columns([1.1, 0.9], gap="large")
        with preview_col:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(f"{width} x {height} px. Coordinates start at the top-left corner.")
        with coords_col:
            with st.container(border=True):
                st.markdown("**Reference object endpoints**")
                st.caption("Enter pixel coordinates for both ends of the known-length object.")
                point_one, point_two = st.columns(2)
                with point_one:
                    st.markdown("**Point 1**")
                    st.number_input("x", min_value=0.0, max_value=float(width), value=min(100.0, float(width)), key="ref_x1")
                    st.number_input("y", min_value=0.0, max_value=float(height), value=min(100.0, float(height)), key="ref_y1")
                with point_two:
                    st.markdown("**Point 2**")
                    st.number_input("x ", min_value=0.0, max_value=float(width), value=min(200.0, float(width)), key="ref_x2")
                    st.number_input("y ", min_value=0.0, max_value=float(height), value=min(100.0, float(height)), key="ref_y2")
        points = (
            (float(st.session_state.ref_x1), float(st.session_state.ref_y1)),
            (float(st.session_state.ref_x2), float(st.session_state.ref_y2)),
        )
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
        st.caption(f"Photo size: {image.shape[1]} x {image.shape[0]} px")

    analysis_key = repr((image_hash, calibration, reference_mm, points))
    if st.session_state.get("analysis_key") != analysis_key:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)
        st.session_state.pop("saved_case_id", None)

    if st.button("Analyze photo", type="primary", key="analyze_photo"):
        try:
            with st.spinner("Finding facial landmarks..."):
                result = analyze_image(
                    image,
                    "Known reference object" if calibration == "Known reference object" else "Iris estimate",
                    reference_mm,
                    points,
                )
            st.session_state.analysis_key = analysis_key
            st.session_state.analysis_result = result
            st.session_state.pop("analysis_error", None)
            if result is None:
                st.session_state.analysis_error = (
                    "No face detected. Try a clear, front-facing photo with the full face in frame."
                )
        except (ValueError, RuntimeError) as exc:
            st.session_state.analysis_key = analysis_key
            st.session_state.analysis_result = None
            st.session_state.analysis_error = str(exc)

    if st.session_state.get("analysis_key") == analysis_key:
        if st.session_state.get("analysis_error"):
            st.error(st.session_state.analysis_error)
        result = st.session_state.get("analysis_result")
        if result:
            st.divider()
            render_analysis_result(image, result)
    elif calibration == "Known reference object":
        st.caption("Review the endpoints, then select Analyze photo when ready.")


def render_records_tab() -> None:
    """Render saved anonymous measurements and export controls."""
    st.markdown('<div class="section-kicker">Saved research records</div>', unsafe_allow_html=True)
    records = load_records()
    if records.empty:
        st.info("No saved measurements yet. Analyze a photo and choose Save this measurement to add a record.")
        return
    st.dataframe(records, width="stretch", hide_index=True)
    numeric = records.select_dtypes(include="number")
    if not numeric.empty:
        with st.expander("Summary statistics"):
            st.dataframe(numeric.describe().T, width="stretch")
    st.download_button(
        "Download CSV", data=export_csv(), file_name="airwayscope_records.csv", mime="text/csv"
    )


def render_about_tab() -> None:
    """Explain the demo's purpose, methods, limitations, and data handling."""
    st.markdown('<div class="section-kicker">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        "## Experimental facial measurements from a photo\n"
        "AirwayScope is an academic research demonstration. It uses estimated face landmarks "
        "to calculate a small set of facial distances and ratios. It is not clinically validated "
        "and must not be used to diagnose or guide treatment."
    )
    purpose_col, method_col = st.columns(2, gap="large")
    with purpose_col:
        with st.container(border=True):
            st.markdown("### What it measures")
            st.markdown(
                "- **Mouth opening:** inner upper to lower lip distance.\n"
                "- **Mouth width:** distance between mouth corners.\n"
                "- **Jaw width:** distance between left and right jaw landmarks.\n"
                "- **Face height proxy:** forehead proxy to chin.\n"
                "- **Opening-to-height ratio:** mouth opening divided by the face height proxy."
            )
    with method_col:
        with st.container(border=True):
            st.markdown("### How estimates work")
            st.markdown(
                "MediaPipe Face Mesh estimates facial landmarks. The app calculates Euclidean "
                "pixel distances, then applies a scale to express linear measurements in millimetres. "
                "A known reference object is preferred. Iris scaling uses an assumed average iris "
                "diameter of 11.7 mm and is approximate (about 10%)."
            )
    st.markdown("### Limitations")
    st.markdown(
        "Landmarks can shift with pose, perspective, lighting, expression, and occlusion. "
        "Quality messages are simple heuristics, not validated acceptance criteria. "
        "The face-height value is a simplified proxy, not anatomical total face height."
    )
    st.markdown("### Privacy and research use")
    st.markdown(
        "Photos are processed in memory and are not saved. If you save a record, the local SQLite "
        "database stores an anonymous case code, measurements, calibration label, warnings, and "
        "timestamp. You can download saved records as CSV from the Saved records tab. Obtain "
        "informed consent, avoid identifying details, and follow your institution's research "
        "ethics and data-governance requirements."
    )


def main() -> None:
    """Render the AirwayScope analysis and records views."""
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">A</div><div><div class="brand-name">AirwayScope</div>'
        '<div class="brand-sub">Facial measurement research demo</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><div><div class="eyebrow">Photo-based measurement</div><h1>A clearer view of each measurement.</h1>'
        '<p>Add a face photo, review the image and scale, then choose when to analyze it.</p></div>'
        '<div class="demo-pill">RESEARCH DEMO - NOT FOR CLINICAL USE</div></div>',
        unsafe_allow_html=True,
    )
    analysis_tab, records_tab, about_tab = st.tabs(["Measure a photo", "Saved records", "About"])
    with analysis_tab:
        render_analysis_tab()
    with records_tab:
        render_records_tab()
    with about_tab:
        render_about_tab()
    st.markdown(
        '<div class="helper" style="text-align:center;margin-top:30px">Experimental estimates only. Not clinically validated and not for diagnosis or treatment.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
