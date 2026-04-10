"""Finger Counter Pro — core finger-state detection and counting.

This module is intentionally self-contained and testable:
all public functions take plain data (coordinates, handedness)
and return plain results (bools, ints, dataclass).
"""

from __future__ import annotations

import dataclasses

from hand_detector import (
    FINGER_PIPS,
    FINGER_TIPS,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandResult,
)

# Finger names (index into fingers_up list)
FINGER_NAMES: list[str] = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


@dataclasses.dataclass
class FingerState:
    """Per-hand finger analysis."""

    handedness: str          # "Left" or "Right"
    fingers_up: list[bool]   # [thumb, index, middle, ring, pinky]
    finger_count: int        # 0-5
    confidence: float        # detection confidence

    @property
    def names_up(self) -> list[str]:
        """Names of extended fingers."""
        return [n for n, up in zip(FINGER_NAMES, self.fingers_up) if up]


@dataclasses.dataclass
class FrameCount:
    """Aggregated finger count for the entire frame."""

    per_hand: list[FingerState]   # one per detected hand
    total: int                     # sum of all hands (0-10)

    @property
    def hand_count(self) -> int:
        return len(self.per_hand)


# ---------------------------------------------------------------------------
# Pure functions — easy to unit-test
# ---------------------------------------------------------------------------

def is_finger_extended(
    tip_y: float,
    pip_y: float,
    margin: float = 0.02,
) -> bool:
    """Return True if a non-thumb finger's tip is above its PIP joint.

    All values are normalised y-coordinates (0 = top, 1 = bottom).
    """
    return tip_y < pip_y - margin


def is_thumb_extended(
    tip_x: float,
    ip_x: float,
    handedness: str,
) -> bool:
    """Return True if the thumb tip has moved outward past the IP joint.

    MediaPipe mirrors handedness labels (the label describes the hand
    as seen from the camera's perspective), so:
    - "Right" label → user's right hand → thumb extends to the LEFT
      (tip.x < ip.x)
    - "Left"  label → user's left hand  → thumb extends to the RIGHT
      (tip.x > ip.x)
    """
    if handedness == "Right":
        return tip_x < ip_x
    return tip_x > ip_x


def detect_fingers(
    hand: HandResult,
    margin: float = 0.02,
) -> list[bool]:
    """Return ``[thumb, index, middle, ring, pinky]`` extended states."""
    thumb = is_thumb_extended(
        hand.norm(THUMB_TIP)[0],
        hand.norm(THUMB_IP)[0],
        hand.handedness,
    )
    fingers = [thumb]
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(
            is_finger_extended(
                hand.norm(tip_idx)[1],
                hand.norm(pip_idx)[1],
                margin,
            )
        )
    return fingers


def count_fingers(fingers_up: list[bool]) -> int:
    """Count the number of extended fingers."""
    return sum(fingers_up)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

class FingerCounter:
    """Stateless analyser — call :meth:`analyse` per hand."""

    def __init__(self, finger_up_margin: float = 0.02) -> None:
        self._margin = finger_up_margin

    def analyse(self, hand: HandResult) -> FingerState:
        """Analyse a single hand and return finger state."""
        fingers = detect_fingers(hand, self._margin)
        return FingerState(
            handedness=hand.handedness,
            fingers_up=fingers,
            finger_count=count_fingers(fingers),
            confidence=hand.score,
        )

    def analyse_frame(self, hands: list[HandResult]) -> FrameCount:
        """Analyse all hands in one frame."""
        states = [self.analyse(h) for h in hands]
        return FrameCount(
            per_hand=states,
            total=sum(s.finger_count for s in states),
        )
