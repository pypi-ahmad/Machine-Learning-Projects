"""Sign Language Alphabet Recognizer — overlay renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from controller import PredictionResult
from hand_detector import HandDetector

# Colours (BGR)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_YELLOW = (0, 255, 255)
_BG = (40, 40, 40)


def draw_overlay(
    frame: np.ndarray,
    result: PredictionResult,
    detector: HandDetector,
    *,
    show_landmarks: bool = True,
    show_prediction: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Render all overlays onto *frame* (in-place)."""
    import cv2

    h, w = frame.shape[:2]

    # Hand skeleton
    if show_landmarks and result.hand is not None:
        detector.draw_landmarks(frame, result.hand)

    # Prediction label
    if show_prediction and result.smoothed_label:
        _draw_prediction_banner(
            frame, result.smoothed_label, result.confidence,
            show_confidence=show_confidence,
        )

    # "No hand" warning
    if result.hand is None:
        cv2.putText(
            frame, "No hand detected", (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED, 1, cv2.LINE_AA,
        )

    return frame


def _draw_prediction_banner(
    frame: np.ndarray,
    label: str,
    confidence: float,
    *,
    show_confidence: bool = True,
) -> None:
    import cv2

    # Semi-transparent dark banner at top-left
    overlay = frame.copy()
    bw, bh = 200, 90
    cv2.rectangle(overlay, (10, 10), (10 + bw, 10 + bh), _BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Large letter
    cv2.putText(
        frame, label, (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, _GREEN, 4, cv2.LINE_AA,
    )

    # Confidence
    if show_confidence:
        conf_txt = f"{confidence:.0%}"
        cv2.putText(
            frame, conf_txt, (120, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _YELLOW, 1, cv2.LINE_AA,
        )

    # "Sign" subtitle
    cv2.putText(
        frame, "Sign", (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
    )
