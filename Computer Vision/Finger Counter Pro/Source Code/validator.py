"""Finger Counter Pro -- input quality validator."""

from __future__ import annotations

import dataclasses

from hand_detector import MultiHandResult


@dataclasses.dataclass
class ValidationReport:
    """Quality-check results for one frame."""

    ok: bool
    warnings: list[str]


class FrameValidator:
    """Lightweight quality checks on detection output."""

    def validate(self, result: MultiHandResult) -> ValidationReport:
        warnings: list[str] = []
        if result.count == 0:
            warnings.append("No hand detected")
        return ValidationReport(ok=len(warnings) == 0, warnings=warnings)
