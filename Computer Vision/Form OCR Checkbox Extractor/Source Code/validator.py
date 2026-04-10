"""Validation rules for Form OCR Checkbox Extractor.

Checks extracted form fields and checkbox states against
configurable rules and emits warnings for missing or
suspicious values.

Usage::

    from validator import FormValidator
    from config import FormCheckboxConfig

    validator = FormValidator(FormCheckboxConfig())
    report = validator.validate(parse_result)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FormCheckboxConfig
from parser import FormParseResult


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
    text_fields_found: int = 0
    checkboxes_found: int = 0
    checked_count: int = 0
    fields_missing: list[str] = field(default_factory=list)
    confidence_flags: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Text fields: {self.text_fields_found}",
            f"Checkboxes: {self.checkboxes_found} ({self.checked_count} checked)",
        ]
        if self.fields_missing:
            lines.append(f"Missing: {', '.join(self.fields_missing)}")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class FormValidator:
    """Validate parsed form fields against configurable rules."""

    def __init__(self, cfg: FormCheckboxConfig) -> None:
        self.cfg = cfg

    def validate(self, result: FormParseResult) -> ValidationReport:
        report = ValidationReport()
        report.text_fields_found = len(result.text_fields)
        report.checkboxes_found = result.num_checkboxes
        report.checked_count = result.num_checked

        self._check_missing(result, report)
        self._check_confidence(result, report)
        self._check_unlabelled_checkboxes(result, report)
        self._check_no_checkboxes(result, report)

        if any(w.severity == "error" for w in report.warnings):
            report.valid = False

        return report

    # ── rules ─────────────────────────────────────────────

    def _check_missing(
        self, result: FormParseResult, report: ValidationReport,
    ) -> None:
        if not self.cfg.warn_missing_fields:
            return
        for fname in self.cfg.required_fields:
            if fname not in result.text_fields:
                report.fields_missing.append(fname)
                report.warnings.append(ValidationWarning(
                    field_name=fname,
                    message=f"Required field '{fname}' not found",
                    severity="error",
                ))

    def _check_confidence(
        self, result: FormParseResult, report: ValidationReport,
    ) -> None:
        threshold = self.cfg.confidence_threshold
        for name, tf in result.text_fields.items():
            if tf.confidence < threshold:
                report.confidence_flags.append(name)
                report.warnings.append(ValidationWarning(
                    field_name=name,
                    message=f"Low OCR confidence ({tf.confidence:.2f})",
                    severity="warning",
                ))
        for cb in result.checkbox_fields:
            if cb.label and cb.confidence < threshold:
                report.confidence_flags.append(cb.label[:30])
                report.warnings.append(ValidationWarning(
                    field_name=cb.label[:30],
                    message=f"Low label confidence ({cb.confidence:.2f})",
                    severity="warning",
                ))

    def _check_unlabelled_checkboxes(
        self, result: FormParseResult, report: ValidationReport,
    ) -> None:
        unlabelled = sum(1 for cb in result.checkbox_fields if not cb.label)
        if unlabelled > 0:
            report.warnings.append(ValidationWarning(
                field_name="checkboxes",
                message=f"{unlabelled} checkbox(es) have no associated text label",
                severity="warning",
            ))

    def _check_no_checkboxes(
        self, result: FormParseResult, report: ValidationReport,
    ) -> None:
        if result.num_checkboxes == 0:
            report.warnings.append(ValidationWarning(
                field_name="checkboxes",
                message="No checkboxes or radio buttons detected in this form",
                severity="warning",
            ))
