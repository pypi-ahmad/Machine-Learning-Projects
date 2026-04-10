"""Quality-check validator for Gesture Controlled Slideshow."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GestureConfig
from controller import ControllerResult


@dataclass
class Warning:
    code: str
    message: str


@dataclass
class ValidationReport:
    ok: bool = True
    warnings: list[Warning] = field(default_factory=list)


class GestureValidator:
    """Validate controller results."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg

    def validate(self, result: ControllerResult) -> ValidationReport:
        report = ValidationReport()

        if self.cfg.warn_no_hand and not result.hand_detected:
            report.warnings.append(
                Warning("NO_HAND", "No hand detected in frame"),
            )
            report.ok = False

        return report
