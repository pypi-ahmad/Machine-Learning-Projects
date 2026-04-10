"""Logo Retrieval Brand Match — input validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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


def validate_source(source: str) -> ValidationReport:
    """Validate a query source (image or directory)."""
    report = ValidationReport()
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
    else:
        report.fail(f"Unsupported file type: {p.suffix}")

    return report
