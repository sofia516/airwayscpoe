"""Image loading and input validation."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def load_image(file: BinaryIO) -> np.ndarray:
    """Read an uploaded image or webcam file-like object as an EXIF-corrected BGR array.

    Args:
        file: Binary stream supplied by Streamlit, such as UploadedFile.
    Returns:
        Image pixels in OpenCV-compatible BGR order.
    Raises:
        ValueError: If the file is not a supported image or is too small.
    """
    try:
        raw = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
        with Image.open(BytesIO(raw)) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            array = np.asarray(rgb)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Could not read this image. Use a valid JPEG, PNG, or WebP file.") from exc
    if min(array.shape[:2]) < 480:
        raise ValueError("Image is too small: the short side must be at least 480 pixels.")
    return array[:, :, ::-1].copy()

