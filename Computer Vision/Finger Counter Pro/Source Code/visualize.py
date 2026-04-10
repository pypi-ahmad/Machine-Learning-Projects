"""Finger Counter Pro — overlay renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from finger_counter import FINGER_NAMES, FingerState
from hand_detector import HandDetector, HandResult, MultiHandResult


# Colours (BGR)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_YELLOW = (0, 255, 255)
_BG = (40, 40, 40)


def draw_overlay(
    frame: np.ndarray,
    multi: MultiHandResult,
    per_hand: list[FingerState],
    smoothed_per_hand: dict[str, int],
    smoothed_total: int,
    detector: HandDetector,
    *,
    show_landmarks: bool = True,
    show_finger_state: bool = True,
    show_count: bool = True,
    show_handedness: bool = True,
) -> np.ndarray:
    """Render all overlays onto *frame* (in-place)."""
    import cv2

    # Draw hand skeletons
    if show_landmarks:
        for hand in multi.hands:
            detector.draw_landmarks(frame, hand)

    # Per-hand annotations
    for i, (hand, state) in enumerate(zip(multi.hands, per_hand)):
        _draw_hand_info(
            frame,
            hand,
            state,
            smoothed_per_hand.get(state.handedness, state.finger_count),
            show_finger_state=show_finger_state,
            show_count=show_count,
            show_handedness=show_handedness,
        )

    # Total count banner
    if show_count:
        _draw_total(frame, smoothed_total, len(per_hand))

    return frame


def _draw_hand_info(
    frame: np.ndarray,
    hand: HandResult,
    state: FingerState,
    smoothed: int,
    *,
    show_finger_state: bool,
    show_count: bool,
    show_handedness: bool,
) -> None:
    import cv2

    wrist_x, wrist_y = hand.pixel(0)
    y0 = max(wrist_y - 40, 20)

    # Handedness label
    if show_handedness:
        label = f"{state.handedness} hand"
        cv2.putText(
            frame, label, (wrist_x - 30, y0 - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA,
        )

    # Per-hand count
    if show_count:
        txt = f"{smoothed}"
        cv2.putText(
            frame, txt, (wrist_x - 10, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, _GREEN, 3, cv2.LINE_AA,
        )

    # Finger-state indicators
    if show_finger_state:
        for j, (name, up) in enumerate(zip(FINGER_NAMES, state.fingers_up)):
            col = _GREEN if up else _RED
            cv2.putText(
                frame,
                f"{name[0]}",
                (wrist_x - 30 + j * 22, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )


def _draw_total(
    frame: np.ndarray,
    total: int,
    hand_count: int,
) -> None:
    import cv2

    h, w = frame.shape[:2]
    label = f"Total: {total}"
    sub = f"({hand_count} hand{'s' if hand_count != 1 else ''} detected)"
    # Semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 220, 10), (w - 10, 80), _BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame, label, (w - 210, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, _WHITE, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, sub, (w - 210, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _YELLOW, 1, cv2.LINE_AA,
    )
