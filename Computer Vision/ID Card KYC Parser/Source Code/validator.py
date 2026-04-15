"""Validation rules for ID Card KYC Parser.

Checks extracted fields against configurable rules and emits
warnings for missing or suspicious values.

Usage::

    from validator import IDCardValidator
    from config import IDCardConfig
    from parser import ParseResult

    validator = IDCardValidator(IDCardConfig())
    report = validator.validate(parse_result)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import IDCardConfig
from parser import ParseResult


@dataclass
class ValidationWarning:
    """Single validation warning."""

    field_name: str
    message: str
    severity: str = "warning"  # "warning" | "error"


@dataclass
class ValidationReport:
    """Result of validator.validate()."""

    valid: bool = True
    warnings: list[ValidationWarning] = field(default_factory=list)
    fields_found: int = 0
    fields_missing: list[str] = field(default_factory=list)
    confidence_flags: list[str] = field(default_factory=list)
    card_detected: bool = False

    def summary(self) -> str:
        lines = [f"Fields found: {self.fields_found}"]
        if self.fields_missing:
            lines.append(f"Missing: {', '.join(self.fields_missing)}")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class IDCardValidator:
    """Validate parsed ID card fields against configurable rules."""

    def __init__(self, cfg: IDCardConfig) -> None:
        self.cfg = cfg

    def validate(
        self,
        result: ParseResult,
        card_detected: bool = False,
    ) -> ValidationReport:
        report = ValidationReport()
        report.fields_found = len(result.fields)
        report.card_detected = card_detected

        if not card_detected and self.cfg.detect_card:
            report.warnings.append(ValidationWarning(
                field_name="card_boundary",
                message="No card boundary detected -- OCR ran on full image",
                severity="warning",
            ))

        self._check_missing(result, report)
        self._check_confidence(result, report)
        self._check_date_format(result, report)
        self._check_id_format(result, report)

        if any(w.severity == "error" for w in report.warnings):
            report.valid = False

        return report

    # -- rules ---------------------------------------------------------

    def _check_missing(self, result: ParseResult, report: ValidationReport) -> None:
        if not self.cfg.warn_missing_fields:
            return
        for fname in self.cfg.required_fields:
            if fname not in result.fields:
                report.fields_missing.append(fname)
                sev = "error" if fname in ("full_name", "id_number") else "warning"
                report.warnings.append(ValidationWarning(
                    field_name=fname,
                    message=f"Required field '{fname}' not found",
                    severity=sev,
                ))

    def _check_confidence(
        self,
        result: ParseResult,
        report: ValidationReport,
        threshold: float = 0.55,
    ) -> None:
        for name, ef in result.fields.items():
            if ef.confidence < threshold:
                report.confidence_flags.append(name)
                report.warnings.append(ValidationWarning(
                    field_name=name,
                    message=f"Low OCR confidence ({ef.confidence:.2f})",
                    severity="warning",
                ))

    def _check_date_format(self, result: ParseResult, report: ValidationReport) -> None:
        date_fields = ("date_of_birth", "expiry_date", "issue_date")
        for dname in date_fields:
            ef = result.fields.get(dname)
            if ef is None:
                continue
            val = ef.value
            if not re.match(
                r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$|"
                r"\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$|"
                r"\w+\s+\d{1,2},?\s+\d{4}$|"
                r"\d{1,2}\s+\w+\s+\d{4}$",
                val,
            ):
                report.warnings.append(ValidationWarning(
                    field_name=dname,
                    message=f"Unusual date format: '{val}'",
                    severity="warning",
                ))

    def _check_id_format(self, result: ParseResult, report: ValidationReport) -> None:
        ef = result.fields.get("id_number")
        if ef is None:
            return
        alphanum = re.sub(r"[^A-Za-z0-9]", "", ef.value)
        if len(alphanum) < 4:
            report.warnings.append(ValidationWarning(
                field_name="id_number",
                message=f"ID number suspiciously short ({len(alphanum)} chars): '{ef.value}'",
                severity="warning",
            ))
