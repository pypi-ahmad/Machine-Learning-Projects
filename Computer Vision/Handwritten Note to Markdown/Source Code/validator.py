"""Validation rules for Handwritten Note to Markdown.

Checks recognised lines against configurable rules and emits
warnings for low confidence or empty results.

Usage::

    from validator import NoteValidator
    from config import NoteConfig

    validator = NoteValidator(NoteConfig())
    report = validator.validate(parse_result)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import NoteConfig
from parser import NoteParseResult


@dataclass
class ValidationWarning:
    """Single validation warning."""

    field_name: str
    message: str
    severity: str = "warning"    # "warning" | "error"


@dataclass
class ValidationReport:
    """Result of validator.validate()."""

    valid: bool = True
    warnings: list[ValidationWarning] = field(default_factory=list)
    lines_total: int = 0
    lines_nonempty: int = 0
    low_confidence_lines: int = 0
    mean_confidence: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Lines: {self.lines_nonempty}/{self.lines_total} non-empty",
            f"Mean confidence: {self.mean_confidence:.2f}",
        ]
        if self.low_confidence_lines:
            lines.append(f"Low confidence: {self.low_confidence_lines} line(s)")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class NoteValidator:
    """Validate recognised note text against configurable rules."""

    def __init__(self, cfg: NoteConfig) -> None:
        self.cfg = cfg

    def validate(self, result: NoteParseResult) -> ValidationReport:
        report = ValidationReport()
        report.lines_total = result.num_lines
        report.mean_confidence = result.mean_confidence

        nonempty = [ln for ln in result.lines if ln.text.strip()]
        report.lines_nonempty = len(nonempty)

        self._check_empty(result, report)
        self._check_confidence(result, report)
        self._check_short_lines(result, report)

        if any(w.severity == "error" for w in report.warnings):
            report.valid = False

        return report

    # ── rules ─────────────────────────────────────────────

    def _check_empty(
        self, result: NoteParseResult, report: ValidationReport,
    ) -> None:
        if not result.lines or all(
            not ln.text.strip() for ln in result.lines
        ):
            report.warnings.append(ValidationWarning(
                field_name="content",
                message="No text recognised in the image",
                severity="error",
            ))

    def _check_confidence(
        self, result: NoteParseResult, report: ValidationReport,
    ) -> None:
        if not self.cfg.warn_low_confidence:
            return
        threshold = self.cfg.confidence_threshold
        low = 0
        for ln in result.lines:
            if ln.text.strip() and ln.confidence < threshold:
                low += 1
                report.warnings.append(ValidationWarning(
                    field_name=f"line_{ln.line_index}",
                    message=(
                        f"Low confidence ({ln.confidence:.2f}): "
                        f"'{ln.text[:40]}'"
                    ),
                    severity="warning",
                ))
        report.low_confidence_lines = low

    def _check_short_lines(
        self, result: NoteParseResult, report: ValidationReport,
    ) -> None:
        for ln in result.lines:
            text = ln.text.strip()
            if text and len(text) < self.cfg.min_text_length:
                report.warnings.append(ValidationWarning(
                    field_name=f"line_{ln.line_index}",
                    message=f"Very short text ({len(text)} char): '{text}'",
                    severity="warning",
                ))
