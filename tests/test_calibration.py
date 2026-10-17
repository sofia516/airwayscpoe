"""Unit tests for scale calibration mathematics."""

import pytest

from airway.calibration import calibrate_scale


def test_reference_calibration_uses_euclidean_distance() -> None:
    """Known length divided by point separation yields mm per pixel."""
    scale, method = calibrate_scale(None, "reference", ((0, 0), (3, 4)), 50)
    assert scale == pytest.approx(10)
    assert method == "reference"


def test_iris_calibration_uses_mean_diameter() -> None:
    """Both iris diameters are averaged against the 11.7 mm assumption."""
    scale, method = calibrate_scale(None, "iris", landmarks={
        "iris_left_diameter_px": 20, "iris_right_diameter_px": 30,
    })
    assert scale == pytest.approx(11.7 / 25)
    assert "approximate" in method


def test_reference_rejects_missing_data() -> None:
    """Incomplete calibration inputs fail with an actionable error."""
    with pytest.raises(ValueError, match="requires two points"):
        calibrate_scale(None, "reference")

