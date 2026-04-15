"""Plant Disease Severity Estimator -- input validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_image(path: str | Path) -> np.ndarray:
    """Read and validate a single image file.
    """Read and validate a single image file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be decoded as an image.
    """
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {p.suffix}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Could not decode image: {p}")
    return img


def collect_images(source: str | Path) -> list[Path]:
    """Collect image paths from a file or directory.
    """Collect image paths from a file or directory.

    Parameters
    ----------
    source : str | Path
        A single image path **or** a directory to scan recursively.

    Returns
    -------
    list[Path]
        Sorted list of image paths.
    """
    """
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return [p]
        raise ValueError(f"Not an image file: {p}")
    if p.is_dir():
        return sorted(
            f for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Source not found: {p}")
