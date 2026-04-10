"""Validation module for Driver Drowsiness Monitor.

Checks pipeline results for common issues:
no face detected, low landmark confidence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DrowsinessConfig
from parser import DrowsinessResult


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


class DrowsinessValidator:
    """Validate drowsiness pipeline results."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg

    def validate(self, result: DrowsinessResult) -> ValidationReport:
        report = ValidationReport()

        if self.cfg.warn_no_face and not result.face_detected:
            report.warnings.append(Warning(
                "face", "No face detected — driver may be out of frame",
            ))

        if report.warnings:
            report.valid = False

        return report
