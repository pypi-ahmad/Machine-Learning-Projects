"""Fire Area Segmentation — input validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationReport:
    ok: bool = True
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.warnings.append(msg)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTS


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS


def validate_source(source: str) -> ValidationReport:
    """Validate a source argument."""
    report = ValidationReport()
    if source.isdigit():
        return report
    p = Path(source)
    if not p.exists():
        report.fail(f"Source not found: {p}")
        return report
    if p.is_dir():
        imgs = [f for f in p.iterdir() if is_image(f)]
        if not imgs:
            report.fail(f"No images found in directory: {p}")
        return report
    ext = p.suffix.lower()
    if ext not in _IMAGE_EXTS and ext not in _VIDEO_EXTS:
        report.warn(f"Unexpected extension '{ext}' — may not be supported")
    return report
