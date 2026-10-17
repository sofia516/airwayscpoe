"""MediaPipe Face Mesh landmark extraction."""

from __future__ import annotations

import cv2
import numpy as np


def detect_landmarks(image: np.ndarray) -> dict | None:
    """Detect the largest face and return named pixel landmark coordinates.

    Face Mesh indices follow MediaPipe's canonical 468-point topology; refined iris
    points 468–477 are used for iris centres. Coordinates are (x, y) pixels.
    Returns None when no face is present. Multiple faces are resolved by choosing
    the face with the largest bounding-box area.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required for facial landmark detection.") from exc
    face_mesh_api = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    if face_mesh_api is None:
        raise RuntimeError(
            "This MediaPipe version does not include the Face Mesh Solutions API. "
            "Install the project dependencies with: python -m pip install -r requirements.txt"
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with face_mesh_api.FaceMesh(
        static_image_mode=True, max_num_faces=5, refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    candidates = []
    for face in result.multi_face_landmarks:
        pts = np.array([(p.x * w, p.y * h) for p in face.landmark], dtype=float)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        candidates.append((float((x1 - x0) * (y1 - y0)), pts))
    pts = max(candidates, key=lambda item: item[0])[1]

    def xy(index: int) -> tuple[float, float]:
        """Return one indexed landmark as (x, y) pixels."""
        return float(pts[index, 0]), float(pts[index, 1])

    def center(indices: tuple[int, ...]) -> tuple[float, float]:
        """Return the mean pixel coordinate of a landmark group."""
        mean = pts[list(indices)].mean(axis=0)
        return float(mean[0]), float(mean[1])

    def diameter(indices: tuple[int, ...]) -> float:
        """Return the mean pairwise distance around an iris boundary."""
        ring = pts[list(indices)]
        return float(np.mean([np.linalg.norm(ring[i] - ring[j]) for i, j in ((0, 2), (1, 3))]))

    # Inner lip contour: upper 13, lower 14; mouth corners: 61 and 291.
    # Chin: 152; jaw extremes: 234 and 454; forehead proxy: 10.
    # Eye corner positions 33/263 give the eye-line tilt; iris rings 468–472/473–477.
    return {
        "inner_upper_lip": xy(13), "inner_lower_lip": xy(14),
        "mouth_left": xy(61), "mouth_right": xy(291), "chin": xy(152),
        "jaw_left": xy(234), "jaw_right": xy(454), "forehead": xy(10),
        "eye_left": xy(33), "eye_right": xy(263),
        "iris_left_center": center((468, 469, 470, 471, 472)),
        "iris_right_center": center((473, 474, 475, 476, 477)),
        "iris_left_diameter_px": diameter((468, 469, 470, 471, 472)),
        "iris_right_diameter_px": diameter((473, 474, 475, 476, 477)),
    }

