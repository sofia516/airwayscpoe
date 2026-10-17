"""Pixel-to-millimetre scale calibration."""

from __future__ import annotations

from math import hypot

import numpy as np


def calibrate_scale(
    image: np.ndarray, method: str, points: tuple[tuple[float, float], tuple[float, float]] | None = None,
    known_length_mm: float | None = None, landmarks: dict | None = None,
) -> tuple[float, str]:
    """Return (millimetres per pixel, method label) for reference or iris calibration.

    Reference calibration divides known real length by the Euclidean distance
    between the user-selected points. Iris calibration uses 11.7 mm divided by
    the average detected horizontal iris diameter, and is approximate (about ±10%).
    """
    if method == "reference":
        if points is None or known_length_mm is None:
            raise ValueError("Reference calibration requires two points and a known length in mm.")
        if known_length_mm <= 0:
            raise ValueError("Reference length must be greater than zero.")
        pixel_distance = hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
        if pixel_distance <= 0:
            raise ValueError("Reference points must be distinct.")
        return known_length_mm / pixel_distance, "reference"
    if method == "iris":
        if landmarks is None:
            raise ValueError("Iris calibration requires detected landmarks.")
        diameters = [landmarks.get("iris_left_diameter_px"), landmarks.get("iris_right_diameter_px")]
        diameters = [float(value) for value in diameters if value is not None and float(value) > 0]
        if not diameters:
            raise ValueError("Could not measure iris diameter for calibration.")
        return 11.7 / (sum(diameters) / len(diameters)), "iris (approximate, ±10%)"
    raise ValueError("Choose 'reference' or 'iris' calibration.")

