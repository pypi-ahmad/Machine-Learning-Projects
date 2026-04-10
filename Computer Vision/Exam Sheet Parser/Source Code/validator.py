"""Validation rules for Exam Sheet Parser.

Checks exam sheet parsing results against configurable quality
rules and emits warnings for low confidence, no questions, or
missing marks.

Usage::

    from validator import ExamSheetValidator
    from config import ExamSheetConfig

    validator = ExamSheetValidator(ExamSheetConfig())
    report = validator.validate(exam_sheet_result)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ExamSheetConfig
from parser import ExamSheetResult


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
    blocks_total: int = 0
    questions_found: int = 0
    low_confidence: int = 0

    def summary(self) -> str:
        lines = [
            f"Blocks: {self.blocks_total}",
            f"Questions: {self.questions_found}",
        ]
        if self.low_confidence:
            lines.append(f"Low confidence: {self.low_confidence}")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class ExamSheetValidator:
    """Validate exam sheet results against quality rules."""

    def __init__(self, cfg: ExamSheetConfig) -> None:
        self.cfg = cfg

    def validate(self, result: ExamSheetResult) -> ValidationReport:
        report = ValidationReport()
        report.blocks_total = result.num_blocks
        report.questions_found = result.num_questions

        self._check_no_blocks(result, report)
        self._check_no_questions(result, report)
        self._check_confidence(result, report)
        self._check_missing_marks(result, report)

        if any(w.severity == "error" for w in report.warnings):
            report.valid = False

        return report

    # ── rules ─────────────────────────────────────────────

    def _check_no_blocks(
        self, result: ExamSheetResult, report: ValidationReport,
    ) -> None:
        if result.num_blocks == 0:
            report.warnings.append(ValidationWarning(
                field_name="ocr_blocks",
                message="No text detected in the document",
                severity="error",
            ))

    def _check_no_questions(
        self, result: ExamSheetResult, report: ValidationReport,
    ) -> None:
        if not self.cfg.warn_no_questions:
            return
        if result.num_questions == 0 and result.num_blocks > 0:
            report.warnings.append(ValidationWarning(
                field_name="questions",
                message="No numbered questions detected — "
                        "document may not be an exam sheet",
                severity="warning",
            ))

    def _check_confidence(
        self, result: ExamSheetResult, report: ValidationReport,
    ) -> None:
        if not self.cfg.warn_low_confidence:
            return
        threshold = self.cfg.confidence_threshold
        low = 0
        for q in result.questions:
            if q.confidence < threshold:
                low += 1
                report.warnings.append(ValidationWarning(
                    field_name=f"Q{q.number}",
                    message=f"Low OCR confidence ({q.confidence:.2f})",
                    severity="warning",
                ))
        report.low_confidence = low

    def _check_missing_marks(
        self, result: ExamSheetResult, report: ValidationReport,
    ) -> None:
        missing = [q for q in result.questions if q.marks is None]
        if missing and result.num_questions > 0:
            nums = ", ".join(f"Q{q.number}" for q in missing[:5])
            suffix = f" (+{len(missing)-5} more)" if len(missing) > 5 else ""
            report.warnings.append(ValidationWarning(
                field_name="marks",
                message=f"No marks found for: {nums}{suffix}",
                severity="warning",
            ))
