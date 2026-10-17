"""Image and pose quality checks."""

from __future__ import annotations

import cv2
import numpy as np


def validate_quality(image: np.ndarray, landmarks: dict) -> list[str]:
    """Return review warnings for pose, face scale, mouth opening, blur, and light.

    Head tilt is the absolute angle from horizontal between Face Mesh eye corners
    (33 and 263). Face height uses forehead proxy 10 to chin 152. Blur is flagged
    below Laplacian variance 100; low light is mean grayscale below 50 on 0–255.
    Thresholds are demo heuristics and are not clinically validated.
    """
    warnings = []
    eye_a, eye_b = landmarks["eye_left"], landmarks["eye_right"]
    angle = abs(float(np.degrees(np.arctan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))))
    if angle > 90:
        angle = abs(180 - angle)
    if angle > 10:
        warnings.append("Head tilt is over 10°; reposition the head level.")
    h, w = image.shape[:2]
    face_height = abs(float(landmarks["chin"][1]) - float(landmarks["forehead"][1]))
    if face_height < 0.25 * h:
        warnings.append("Face is smaller than 25% of image height; move closer.")
    mouth_gap = np.linalg.norm(np.asarray(landmarks["inner_upper_lip"]) - np.asarray(landmarks["inner_lower_lip"]))
    if mouth_gap < max(2.0, 0.005 * h):
        warnings.append("Mouth does not appear open; open as wide as comfortable.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
        warnings.append("Image may be blurry; use a stable camera and refocus.")
    if float(np.mean(gray)) < 50:
        warnings.append("Image may be too dark; use even front lighting.")
    return warnings

