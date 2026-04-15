"""Validation module for Face Verification Attendance System.

Checks pipeline state and results for common issues.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FaceAttendanceConfig
from parser import AttendanceResult


@dataclass
class Warning:
    """Single validation warning."""

    field_name: str
    message: str


@dataclass
class ValidationReport:
    """Pipeline validation report."""

    valid: bool = True
    warnings: list[Warning] = field(default_factory=list)

    def summary(self) -> str:
        if self.valid and not self.warnings:
            return "OK"
        parts = []
        if not self.valid:
            parts.append("INVALID")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return "; ".join(parts)


class AttendanceValidator:
    """Validate pipeline results and configuration state."""

    def __init__(self, cfg: FaceAttendanceConfig) -> None:
        self.cfg = cfg

    def validate(
        self,
        result: AttendanceResult,
        gallery_size: int = 0,
    ) -> ValidationReport:
        """Validate a pipeline result.

        Parameters
        ----------
        result : AttendanceResult
            Pipeline output.
        gallery_size : int
            Number of enrolled identities.

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport()

        # No gallery
        if self.cfg.warn_no_gallery and gallery_size == 0:
            report.warnings.append(Warning(
                "gallery",
                "No identities enrolled -- all faces will be Unknown",
            ))

        # No faces detected
        if self.cfg.warn_no_faces and result.num_faces == 0:
            report.warnings.append(Warning(
                "faces", "No faces detected in frame",
            ))

        # Low confidence faces
        if self.cfg.warn_low_confidence and result.matches:
            low = [
                m for m in result.matches
                if m.det_confidence < self.cfg.confidence_threshold
            ]
            if low:
                report.warnings.append(Warning(
                    "confidence",
                    f"{len(low)} face(s) below confidence threshold "
                    f"({self.cfg.confidence_threshold:.2f})",
                ))

        # All unknown
        if result.num_faces > 0 and result.num_matched == 0 and gallery_size > 0:
            report.warnings.append(Warning(
                "matching",
                "No faces matched any enrolled identity",
            ))

        if report.warnings:
            report.valid = False

        return report
