"""Building Footprint Change Detector — input validation."""

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


def validate_image_path(path: str | Path) -> ValidationReport:
    """Check that *path* points to a readable image file."""
    report = ValidationReport()
    p = Path(path)
    if not p.exists():
        report.fail(f"File not found: {p}")
        return report
    if p.suffix.lower() not in _IMAGE_EXTS:
        report.warn(f"Unexpected extension '{p.suffix}' -- may not be an image")
    return report


def validate_pair(before: str | Path, after: str | Path) -> ValidationReport:
    """Validate a before/after image pair."""
    report = ValidationReport()
    for label, path in [("before", before), ("after", after)]:
        r = validate_image_path(path)
        if not r.ok:
            report.fail(f"[{label}] {r.warnings[0]}")
        else:
            for w in r.warnings:
                report.warn(f"[{label}] {w}")
    return report


def validate_directory_pair(
    before_dir: str | Path,
    after_dir: str | Path,
) -> tuple[ValidationReport, list[tuple[Path, Path]]]:
    """Match image files in two directories by filename.

    Returns a report plus a list of ``(before_path, after_path)`` pairs.
    """
    report = ValidationReport()
    bd = Path(before_dir)
    ad = Path(after_dir)

    if not bd.is_dir():
        report.fail(f"Before directory not found: {bd}")
        return report, []
    if not ad.is_dir():
        report.fail(f"After directory not found: {ad}")
        return report, []

    before_files = {f.name: f for f in bd.iterdir() if f.suffix.lower() in _IMAGE_EXTS}
    after_files = {f.name: f for f in ad.iterdir() if f.suffix.lower() in _IMAGE_EXTS}

    common = sorted(set(before_files) & set(after_files))
    if not common:
        report.fail("No matching filenames found in before/after directories")
        return report, []

    only_before = set(before_files) - set(after_files)
    only_after = set(after_files) - set(before_files)
    if only_before:
        report.warn(f"{len(only_before)} file(s) only in before dir (skipped)")
    if only_after:
        report.warn(f"{len(only_after)} file(s) only in after dir (skipped)")

    pairs = [(before_files[n], after_files[n]) for n in common]
    return report, pairs
