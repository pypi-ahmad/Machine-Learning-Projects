"""Similar Image Finder -- input validation helpers."""

from __future__ import annotations

from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_file(path: str | Path) -> bool:
    """Return True if *path* points to an existing image file."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() in _IMAGE_EXTS


def validate_image(path: str | Path) -> Path:
    """Return validated Path or raise ValueError."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    if p.suffix.lower() not in _IMAGE_EXTS:
        raise ValueError(f"Unsupported image format: {p.suffix}")
    return p


def validate_directory(path: str | Path) -> Path:
    """Return validated directory Path or raise ValueError."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Directory does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")
    return p


def collect_images(directory: str | Path, *, recursive: bool = True) -> list[Path]:
    """Collect all image files from a directory."""
    d = validate_directory(directory)
    pattern = "**/*" if recursive else "*"
    images = sorted(
        p for p in d.glob(pattern)
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    return images


def infer_category(image_path: Path, root: Path) -> str:
    """Infer category from the immediate parent directory name.
    """Infer category from the immediate parent directory name.

    If the parent is the root itself, returns empty string.
    """
    """
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        return ""
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return ""
