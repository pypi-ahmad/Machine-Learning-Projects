"""Food Freshness Grader — input validation."""

from __future__ import annotations

from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_file(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix.lower() in _IMAGE_EXTS


def validate_image(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    if p.suffix.lower() not in _IMAGE_EXTS:
        raise ValueError(f"Unsupported image format: {p.suffix}")
    return p


def validate_directory(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Directory does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")
    return p


def collect_images(directory: str | Path, *, recursive: bool = True) -> list[Path]:
    d = validate_directory(directory)
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in d.glob(pattern)
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
