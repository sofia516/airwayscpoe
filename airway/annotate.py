"""Draw simple landmark and measurement overlays."""

from __future__ import annotations

import cv2
import numpy as np


def annotate_image(image: np.ndarray, landmarks: dict, measurements: dict) -> np.ndarray:
    """Return a copy of the BGR image with facial points and labelled distances in mm."""
    output = image.copy()
    points = {
        "inner_upper_lip": "mouth_opening_mm", "inner_lower_lip": "mouth_opening_mm",
        "mouth_left": "mouth_width_mm", "mouth_right": "mouth_width_mm",
        "jaw_left": "jaw_width_mm", "jaw_right": "jaw_width_mm",
        "forehead": "face_height_mm", "chin": "face_height_mm",
    }
    for key in points:
        x, y = (int(round(v)) for v in landmarks[key])
        cv2.circle(output, (x, y), 4, (0, 220, 255), -1, cv2.LINE_AA)
    lines = [
        ("inner_upper_lip", "inner_lower_lip", "mouth_opening_mm"),
        ("mouth_left", "mouth_right", "mouth_width_mm"),
        ("jaw_left", "jaw_right", "jaw_width_mm"),
        ("forehead", "chin", "face_height_mm"),
    ]
    for start, end, key in lines:
        a = tuple(int(round(v)) for v in landmarks[start])
        b = tuple(int(round(v)) for v in landmarks[end])
        cv2.line(output, a, b, (50, 220, 50), 2, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        label = f"{measurements[key]:.1f} mm"
        cv2.putText(output, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(output, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output

