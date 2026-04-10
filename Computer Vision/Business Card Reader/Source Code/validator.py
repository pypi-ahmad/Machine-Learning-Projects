"""Validation rules for Business Card Reader.

Checks extracted contact fields against configurable rules and emits
warnings for missing or suspicious values.

Usage::

    from validator import CardValidator
    from config import CardConfig
    from parser import ParseResult

    validator = CardValidator(CardConfig())
    report = validator.validate(parse_result)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CardConfig
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

    def summary(self) -> str:
        lines = [f"Fields found: {self.fields_found}"]
        if self.fields_missing:
            lines.append(f"Missing: {', '.join(self.fields_missing)}")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class CardValidator:
    """Validate parsed business card fields against configurable rules."""

    def __init__(self, cfg: CardConfig) -> None:
        self.cfg = cfg

    def validate(self, result: ParseResult) -> ValidationReport:
        report = ValidationReport()
        report.fields_found = len(result.fields)

        self._check_missing(result, report)
        self._check_confidence(result, report)
        self._check_email_format(result, report)
        self._check_phone_format(result, report)

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
                sev = "error" if fname == "name" else "warning"
                report.warnings.append(ValidationWarning(
                    field_name=fname,
                    message=f"Required field '{fname}' not found",
                    severity=sev,
                ))

    def _check_confidence(
        self,
        result: ParseResult,
        report: ValidationReport,
        threshold: float = 0.6,
    ) -> None:
        for name, ef in result.fields.items():
            if ef.confidence < threshold:
                report.confidence_flags.append(name)
                report.warnings.append(ValidationWarning(
                    field_name=name,
                    message=f"Low OCR confidence ({ef.confidence:.2f})",
                    severity="warning",
                ))

    def _check_email_format(self, result: ParseResult, report: ValidationReport) -> None:
        ef = result.fields.get("email")
        if ef is None:
            return
        if not re.match(
            r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
            ef.value,
        ):
            report.warnings.append(ValidationWarning(
                field_name="email",
                message=f"Possibly malformed email: '{ef.value}'",
                severity="warning",
            ))

    def _check_phone_format(self, result: ParseResult, report: ValidationReport) -> None:
        ef = result.fields.get("phone")
        if ef is None:
            return
        digits = re.sub(r"\D", "", ef.value)
        if len(digits) < 7 or len(digits) > 15:
            report.warnings.append(ValidationWarning(
                field_name="phone",
                message=f"Unusual phone digit count ({len(digits)}): '{ef.value}'",
                severity="warning",
            ))
