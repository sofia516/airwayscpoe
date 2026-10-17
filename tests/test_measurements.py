"""Unit tests for measurement maths using synthetic landmark coordinates."""

import pytest

from airway.measurements import measure_airway_features


@pytest.fixture
def synthetic_landmarks() -> dict:
    """Return points with simple exact distances for predictable calculations."""
    return {
        "inner_upper_lip": (0, 0), "inner_lower_lip": (0, 4),
        "mouth_left": (0, 0), "mouth_right": (3, 4),
        "jaw_left": (0, 0), "jaw_right": (6, 8),
        "forehead": (0, 0), "chin": (0, 10),
    }


def test_measurements_and_ratio(synthetic_landmarks: dict) -> None:
    """All lengths scale linearly, and opening ratio uses face height."""
    result = measure_airway_features(synthetic_landmarks, 2.0)
    assert result["mouth_opening_mm"] == pytest.approx(8)
    assert result["mouth_width_mm"] == pytest.approx(10)
    assert result["jaw_width_mm"] == pytest.approx(20)
    assert result["face_height_mm"] == pytest.approx(20)
    assert result["mouth_opening_to_face_height_ratio"] == pytest.approx(0.4)


def test_zero_scale_rejected(synthetic_landmarks: dict) -> None:
    """Scale must be positive to produce meaningful measurements."""
    with pytest.raises(ValueError, match="greater than zero"):
        measure_airway_features(synthetic_landmarks, 0)

