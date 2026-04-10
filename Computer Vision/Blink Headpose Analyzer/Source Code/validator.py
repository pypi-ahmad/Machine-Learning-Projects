"""Quality-check validator for Blink Headpose Analyzer."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzer import AnalysisResult
from config import AnalyzerConfig


@dataclass
class Warning:
    code: str
    message: str


@dataclass
class ValidationReport:
    ok: bool = True
    warnings: list[Warning] = field(default_factory=list)


class AnalyzerValidator:
    """Validate analysis results with configurable rules."""

    def __init__(self, cfg: AnalyzerConfig) -> None:
        self.cfg = cfg

    def validate(self, result: AnalysisResult) -> ValidationReport:
        report = ValidationReport()

        if self.cfg.warn_no_face and not result.face_detected:
            report.warnings.append(
                Warning("NO_FACE", "No face detected in frame"),
            )
            report.ok = False

        return report
