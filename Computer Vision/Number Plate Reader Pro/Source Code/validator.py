"""Validation rules for Number Plate Reader Pro.

Checks plate read results against configurable quality rules
and emits warnings for low confidence or invalid plates.

Usage::

    from validator import PlateValidator
    from config import PlateConfig

    validator = PlateValidator(PlateConfig())
    report = validator.validate(plate_read_result)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PlateConfig
from parser import PlateReadResult


@dataclass
class ValidationWarning:
    """Single validation warning."""

    field_name: str
    message: str
    severity: str = "warning"   # "warning" | "error"


@dataclass
class ValidationReport:
    """Result of validator.validate()."""

    valid: bool = True
    warnings: list[ValidationWarning] = field(default_factory=list)
    plates_total: int = 0
    plates_valid: int = 0
    plates_new: int = 0
    low_confidence: int = 0

    def summary(self) -> str:
        lines = [
            f"Plates: {self.plates_total} detected, "
            f"{self.plates_valid} valid, {self.plates_new} new",
        ]
        if self.low_confidence:
            lines.append(f"Low confidence: {self.low_confidence}")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class PlateValidator:
    """Validate plate read results against quality rules."""

    def __init__(self, cfg: PlateConfig) -> None:
        self.cfg = cfg

    def validate(self, result: PlateReadResult) -> ValidationReport:
        report = ValidationReport()
        report.plates_total = result.num_detections
        report.plates_valid = result.num_valid
        report.plates_new = result.num_new

        self._check_no_plates(result, report)
        self._check_confidence(result, report)
        self._check_invalid_plates(result, report)

        if any(w.severity == "error" for w in report.warnings):
            report.valid = False

        return report

    # ── rules ─────────────────────────────────────────────

    def _check_no_plates(
        self, result: PlateReadResult, report: ValidationReport,
    ) -> None:
        if result.num_detections == 0:
            report.warnings.append(ValidationWarning(
                field_name="detection",
                message="No license plates detected in frame",
                severity="warning",
            ))

    def _check_confidence(
        self, result: PlateReadResult, report: ValidationReport,
    ) -> None:
        threshold = self.cfg.confidence_threshold
        low = 0
        for read in result.reads:
            if read.plate_text and read.ocr_confidence < threshold:
                low += 1
                report.warnings.append(ValidationWarning(
                    field_name=read.plate_text[:15],
                    message=f"Low OCR confidence ({read.ocr_confidence:.2f})",
                    severity="warning",
                ))
        report.low_confidence = low

    def _check_invalid_plates(
        self, result: PlateReadResult, report: ValidationReport,
    ) -> None:
        for read in result.reads:
            if read.plate_text and not read.is_valid:
                report.warnings.append(ValidationWarning(
                    field_name=read.plate_text[:15],
                    message="Plate text does not match expected pattern",
                    severity="warning",
                ))
