"""Yoga Pose Correction Coach -- overlay renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from controller import CoachResult
from pose_detector import PoseDetector

# Colours (BGR)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_YELLOW = (0, 255, 255)
_CYAN = (255, 255, 0)
_ORANGE = (0, 165, 255)
_BG = (40, 40, 40)

_POSE_DISPLAY = {
    "mountain": "Mountain (Tadasana)",
    "warrior_i": "Warrior I",
    "warrior_ii": "Warrior II",
    "tree": "Tree (Vrksasana)",
    "downward_dog": "Downward Dog",
    "unknown": "Unrecognised",
}


def draw_overlay(
    frame: np.ndarray,
    result: CoachResult,
    detector: PoseDetector,
    *,
    show_skeleton: bool = True,
    show_angles: bool = True,
    show_pose_label: bool = True,
    show_confidence: bool = True,
    show_corrections: bool = True,
) -> np.ndarray:
    """Render all overlays onto *frame* (in-place)."""
    import cv2

    h, w = frame.shape[:2]

    # Skeleton
    if show_skeleton and result.pose.detected:
        detector.draw_skeleton(frame, result.pose)

    if result.classification is not None:
        # Pose label + confidence panel
        if show_pose_label or show_confidence:
            _draw_pose_panel(frame, result, show_confidence=show_confidence)

        # Correction hints panel
        if show_corrections and result.corrections:
            _draw_corrections(frame, result)

    # "No pose" warning
    if not result.pose.detected:
        cv2.putText(
            frame, "No pose detected", (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED, 1, cv2.LINE_AA,
        )

    return frame


def _draw_pose_panel(
    frame: np.ndarray,
    result: CoachResult,
    *,
    show_confidence: bool,
) -> None:
    import cv2

    overlay = frame.copy()
    panel_w, panel_h = 320, 70
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), _BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    pose_name = _POSE_DISPLAY.get(result.smoothed_pose, result.smoothed_pose)
    cv2.putText(
        frame, pose_name, (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, _GREEN, 2, cv2.LINE_AA,
    )

    if show_confidence and result.classification:
        conf_txt = f"Confidence: {result.classification.confidence:.0%}"
        cv2.putText(
            frame, conf_txt, (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _YELLOW, 1, cv2.LINE_AA,
        )


def _draw_corrections(frame: np.ndarray, result: CoachResult) -> None:
    import cv2

    h, w = frame.shape[:2]
    hints = result.corrections
    panel_h = 30 + len(hints) * 28
    panel_w = 400
    y0 = h - panel_h - 10

    # Semi-transparent panel at bottom-left
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y0), (10 + panel_w, h - 10), _BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, "Corrections:", (20, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA,
    )

    for i, hint in enumerate(hints):
        col = _ORANGE if hint.severity == "major" else _YELLOW
        icon = "!" if hint.severity == "major" else "-"
        txt = f"{icon} {hint.joint}: {hint.hint}"
        # Truncate if too long
        if len(txt) > 55:
            txt = txt[:52] + "..."
        cv2.putText(
            frame, txt, (25, y0 + 48 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA,
        )
