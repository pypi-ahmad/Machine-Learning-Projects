"""Validation rules for Receipt Digitizer.

Checks extracted fields against configurable rules and emits
warnings for missing or suspicious values.

Usage::

    from validator import ReceiptValidator
    from config import ReceiptConfig
    from parser import ParseResult

    validator = ReceiptValidator(ReceiptConfig())
    report = validator.validate(parse_result)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ReceiptConfig
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


class ReceiptValidator:
    """Validate parsed receipt fields against configurable rules."""

    def __init__(self, cfg: ReceiptConfig) -> None:
        self.cfg = cfg

    def validate(self, result: ParseResult) -> ValidationReport:
        report = ValidationReport()
        report.fields_found = len(result.fields)

        self._check_missing(result, report)
        self._check_confidence(result, report)
        self._check_total_consistency(result, report)
        self._check_date_format(result, report)

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
                sev = "error" if fname == "total" else "warning"
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

    def _check_total_consistency(
        self,
        result: ParseResult,
        report: ValidationReport,
    ) -> None:
        subtotal = result.fields.get("subtotal")
        tax = result.fields.get("tax")
        total = result.fields.get("total")

        if not (subtotal and tax and total):
            return

        try:
            s = float(subtotal.value.replace(",", ""))
            t = float(tax.value.replace(",", ""))
            tot = float(total.value.replace(",", ""))
        except ValueError:
            return

        expected = s + t
        if abs(expected - tot) > 0.02 * max(tot, 1):
            report.warnings.append(ValidationWarning(
                field_name="total",
                message=(
                    f"Subtotal ({s:.2f}) + Tax ({t:.2f}) = {expected:.2f} "
                    f"but Total is {tot:.2f}"
                ),
                severity="warning",
            ))

    def _check_date_format(
        self,
        result: ParseResult,
        report: ValidationReport,
    ) -> None:
        ef = result.fields.get("date")
        if ef is None:
            return
        val = ef.value
        if not re.match(
            r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$|"
            r"\w+\s+\d{1,2},?\s+\d{4}$",
            val,
        ):
            report.warnings.append(ValidationWarning(
                field_name="date",
                message=f"Unusual date format: '{val}'",
                severity="warning",
            ))
