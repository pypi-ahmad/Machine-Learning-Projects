"""Wildlife Species Retrieval -- input validation & helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_image(path: str | Path) -> np.ndarray:
    """Read and validate a single image file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {p.suffix}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Could not decode image: {p}")
    return img


def collect_images(source: str | Path, *, recursive: bool = True) -> list[Path]:
    """Collect image paths from a file or directory."""
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return [p]
        raise ValueError(f"Not an image file: {p}")
    if p.is_dir():
        gen = p.rglob("*") if recursive else p.iterdir()
        return sorted(
            f for f in gen
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Source not found: {p}")


def infer_species(image_path: Path, root: Path) -> str:
    """Infer species label from the image's parent directory name."""
    try:
        rel = image_path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass
    return image_path.parent.name
