"""Validation rules for Scene Text Reader Translator.

Checks scene text results against configurable quality rules
and emits warnings for low confidence or empty reads.

Usage::

    from validator import SceneTextValidator
    from config import SceneTextConfig

    validator = SceneTextValidator(SceneTextConfig())
    report = validator.validate(scene_text_result)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SceneTextConfig
from parser import SceneTextResult


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
    low_confidence: int = 0

    def summary(self) -> str:
        lines = [f"Text blocks: {self.blocks_total}"]
        if self.low_confidence:
            lines.append(f"Low confidence: {self.low_confidence}")
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.field_name}: {w.message}")
        return "\n".join(lines)


class SceneTextValidator:
    """Validate scene text results against quality rules."""

    def __init__(self, cfg: SceneTextConfig) -> None:
        self.cfg = cfg

    def validate(self, result: SceneTextResult) -> ValidationReport:
        report = ValidationReport()
        report.blocks_total = result.num_blocks

        self._check_no_text(result, report)
        self._check_confidence(result, report)
        self._check_short_text(result, report)

        if any(w.severity == "error" for w in report.warnings):
            report.valid = False

        return report

    # ── rules ─────────────────────────────────────────────

    def _check_no_text(
        self, result: SceneTextResult, report: ValidationReport,
    ) -> None:
        if result.num_blocks == 0:
            report.warnings.append(ValidationWarning(
                field_name="text_blocks",
                message="No text detected in the scene",
                severity="warning",
            ))

    def _check_confidence(
        self, result: SceneTextResult, report: ValidationReport,
    ) -> None:
        if not self.cfg.warn_low_confidence:
            return
        threshold = self.cfg.confidence_threshold
        low = 0
        for read in result.reads:
            if read.confidence < threshold:
                low += 1
                report.warnings.append(ValidationWarning(
                    field_name=read.text[:30],
                    message=f"Low OCR confidence ({read.confidence:.2f})",
                    severity="warning",
                ))
        report.low_confidence = low

    def _check_short_text(
        self, result: SceneTextResult, report: ValidationReport,
    ) -> None:
        for read in result.reads:
            if len(read.text.strip()) < self.cfg.min_text_length:
                report.warnings.append(ValidationWarning(
                    field_name="short_text",
                    message=f"Very short text: '{read.text}'",
                    severity="warning",
                ))
