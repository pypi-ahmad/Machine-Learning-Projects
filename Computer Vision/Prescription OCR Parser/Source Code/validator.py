"""Validation rules for Prescription OCR Parser.

Checks extracted prescription data against configurable rules
and emits warnings for missing or suspicious values.

Usage::

    from validator import PrescriptionValidator
    from config import PrescriptionConfig

    validator = PrescriptionValidator(PrescriptionConfig())
    report = validator.validate(parse_result)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PrescriptionConfig
from parser import PrescriptionResult


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
    medicines_found: int = 0
    fields_missing: list[str] = field(default_factory=list)
    low_confidence_items: int = 0

    def summary(self) -> str:
        lines = [
            f"Medicines found: {self.medicines_found}",
        ]
        if self.fields_missing:
            lines.append(f"Missing: {', '.join(self.fields_missing)}")
        if self.low_confidence_items:
            lines.append(f"Low confidence: {self.low_confidence_items} item(s)")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class PrescriptionValidator:
    """Validate parsed prescription against configurable rules."""

    def __init__(self, cfg: PrescriptionConfig) -> None:
        self.cfg = cfg

    def validate(self, result: PrescriptionResult) -> ValidationReport:
        report = ValidationReport()
        report.medicines_found = result.num_medicines

        self._check_no_medicines(result, report)
        self._check_missing_details(result, report)
        self._check_confidence(result, report)
        self._check_required_header_fields(result, report)

        if any(w.severity == "error" for w in report.warnings):
            report.valid = False

        return report

    # ── rules ─────────────────────────────────────────────

    def _check_no_medicines(
        self, result: PrescriptionResult, report: ValidationReport,
    ) -> None:
        if result.num_medicines == 0:
            report.warnings.append(ValidationWarning(
                field_name="medicines",
                message="No medicine entries detected in the prescription",
                severity="error",
            ))

    def _check_missing_details(
        self, result: PrescriptionResult, report: ValidationReport,
    ) -> None:
        for i, med in enumerate(result.medicines):
            label = med.medicine_name[:30] or f"medicine_{i}"
            if not med.dosage:
                report.fields_missing.append(f"{label}:dosage")
                report.warnings.append(ValidationWarning(
                    field_name=label,
                    message="Missing dosage information",
                    severity="warning",
                ))
            if not med.frequency:
                report.fields_missing.append(f"{label}:frequency")
                report.warnings.append(ValidationWarning(
                    field_name=label,
                    message="Missing frequency / timing information",
                    severity="warning",
                ))

    def _check_confidence(
        self, result: PrescriptionResult, report: ValidationReport,
    ) -> None:
        if not self.cfg.warn_low_confidence:
            return
        threshold = self.cfg.confidence_threshold
        low = 0
        for med in result.medicines:
            if med.confidence < threshold:
                low += 1
                report.warnings.append(ValidationWarning(
                    field_name=med.medicine_name[:30],
                    message=f"Low OCR confidence ({med.confidence:.2f})",
                    severity="warning",
                ))
        for name, ef in result.header_fields.items():
            if ef.confidence < threshold:
                low += 1
                report.warnings.append(ValidationWarning(
                    field_name=name,
                    message=f"Low OCR confidence ({ef.confidence:.2f})",
                    severity="warning",
                ))
        report.low_confidence_items = low

    def _check_required_header_fields(
        self, result: PrescriptionResult, report: ValidationReport,
    ) -> None:
        for fname in self.cfg.required_fields:
            # Check both header fields and medicine names
            if fname == "medicine_name":
                if result.num_medicines == 0:
                    report.fields_missing.append(fname)
                    # Already reported in _check_no_medicines
            elif fname not in result.header_fields:
                report.fields_missing.append(fname)
                report.warnings.append(ValidationWarning(
                    field_name=fname,
                    message=f"Required field '{fname}' not found",
                    severity="warning",
                ))
