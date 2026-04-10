"""Quality-check validator for Gaze Direction Estimator."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzer import GazeAnalysisResult
from config import GazeConfig


@dataclass
class Warning:
    code: str
    message: str


@dataclass
class ValidationReport:
    ok: bool = True
    warnings: list[Warning] = field(default_factory=list)


class GazeValidator:
    """Validate gaze analysis results."""

    def __init__(self, cfg: GazeConfig) -> None:
        self.cfg = cfg

    def validate(self, result: GazeAnalysisResult) -> ValidationReport:
        report = ValidationReport()

        if self.cfg.warn_no_face and not result.face_detected:
            report.warnings.append(
                Warning("NO_FACE", "No face detected in frame"),
            )
            report.ok = False

        if result.face_detected and not result.iris.detected:
            report.warnings.append(
                Warning("NO_IRIS", "Face found but iris not located"),
            )
            report.ok = False

        return report
