"""Sign Language Alphabet Recognizer -- quality validator."""

from __future__ import annotations

import dataclasses

from hand_detector import HandResult


@dataclasses.dataclass
class ValidationReport:
    ok: bool
    warnings: list[str]


class FrameValidator:
    """Lightweight quality checks on detection output."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._min_conf = min_confidence

    def validate(self, hand: HandResult | None) -> ValidationReport:
        warnings: list[str] = []
        if hand is None:
            warnings.append("No hand detected")
        elif hand.score < self._min_conf:
            warnings.append(f"Low detection confidence: {hand.score:.2f}")
        return ValidationReport(ok=len(warnings) == 0, warnings=warnings)
