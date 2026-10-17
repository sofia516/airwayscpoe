"""Facial feature measurements from Face Mesh landmarks."""

from __future__ import annotations

from math import hypot


def measure_airway_features(landmarks: dict, mm_per_px: float) -> dict[str, float]:
    """Compute linear facial measurements and their ratio in physical units.

    All distances are Euclidean pixel distances multiplied by mm_per_px.
    mouth_opening_mm: distance between inner upper/lower lip (Face Mesh 13, 14).
    mouth_width_mm: distance between mouth corners (61, 291).
    jaw_width_mm: distance between left/right jaw points (234, 454).
    face_height_mm: forehead proxy to chin (10, 152); this is not anatomical
    total face height. mouth_opening_to_face_height_ratio is the unrounded ratio
    of the first distance to face_height_mm. Values are descriptive research
    measures, not clinical airway measurements.
    """
    if mm_per_px <= 0:
        raise ValueError("Scale must be greater than zero.")

    def distance(first: str, second: str) -> float:
        """Return scaled Euclidean distance between named points."""
        a, b = landmarks[first], landmarks[second]
        return hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])) * mm_per_px

    opening = distance("inner_upper_lip", "inner_lower_lip")
    height = distance("forehead", "chin")
    if height <= 0:
        raise ValueError("Face height is zero; check the detected landmarks.")
    return {
        "mouth_opening_mm": opening,
        "mouth_width_mm": distance("mouth_left", "mouth_right"),
        "jaw_width_mm": distance("jaw_left", "jaw_right"),
        "face_height_mm": height,
        "mouth_opening_to_face_height_ratio": opening / height,
    }

