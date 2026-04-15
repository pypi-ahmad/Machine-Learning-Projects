"""Interactive Video Object Cutout Studio -- input validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


@dataclass
class ValidationReport:
    ok: bool = True
    source_type: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.warnings.append(msg)


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTS


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS


def is_webcam(source: str) -> bool:
    try:
        int(source)
        return True
    except (ValueError, TypeError):
        return False


def validate_source(source: str) -> ValidationReport:
    """Determine source type and validate existence."""
    report = ValidationReport()

    if is_webcam(source):
        report.source_type = "webcam"
        return report

    p = Path(source)
    if not p.exists():
        report.fail(f"Source does not exist: {source}")
        return report

    if p.is_dir():
        report.source_type = "directory"
        imgs = [f for f in p.iterdir() if is_image(f)]
        if not imgs:
            report.fail(f"No image files found in: {source}")
        return report

    if is_image(p):
        report.source_type = "image"
    elif is_video(p):
        report.source_type = "video"
    else:
        report.fail(f"Unsupported file type: {p.suffix}")

    return report
