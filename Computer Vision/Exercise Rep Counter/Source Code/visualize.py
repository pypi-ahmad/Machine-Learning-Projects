"""Exercise Rep Counter — overlay renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from controller import ControllerResult
from pose_detector import PoseDetector

# Colours (BGR)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_YELLOW = (0, 255, 255)
_CYAN = (255, 255, 0)
_BG = (40, 40, 40)

_EXERCISE_DISPLAY = {
    "squat": "Squat",
    "pushup": "Push-Up",
    "bicep_curl": "Bicep Curl",
}


def draw_overlay(
    frame: np.ndarray,
    result: ControllerResult,
    detector: PoseDetector,
    *,
    show_skeleton: bool = True,
    show_angles: bool = True,
    show_rep_count: bool = True,
    show_stage: bool = True,
    show_exercise: bool = True,
) -> np.ndarray:
    """Render all overlays onto *frame* (in-place)."""
    import cv2

    h, w = frame.shape[:2]

    # Skeleton
    if show_skeleton and result.pose.detected:
        detector.draw_skeleton(frame, result.pose)

    if result.analysis is not None and result.rep_state is not None:
        # Angle at the joint
        if show_angles:
            _draw_angle_label(frame, result)

        # Rep count + stage banner
        if show_rep_count or show_stage or show_exercise:
            _draw_info_panel(
                frame, result,
                show_rep_count=show_rep_count,
                show_stage=show_stage,
                show_exercise=show_exercise,
            )

    # "No pose" warning
    if not result.pose.detected:
        cv2.putText(
            frame, "No pose detected", (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED, 1, cv2.LINE_AA,
        )

    return frame


def _draw_angle_label(frame: np.ndarray, result: ControllerResult) -> None:
    """Draw the measured angle near the joint vertex."""
    import cv2

    analysis = result.analysis
    pose = result.pose
    _, vertex_idx, _ = analysis.landmarks_used
    vx, vy = pose.pixel(vertex_idx)

    txt = f"{analysis.angle:.0f} deg"
    cv2.putText(
        frame, txt, (vx + 10, vy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _CYAN, 2, cv2.LINE_AA,
    )

    # Draw the three landmarks as coloured circles
    for i, idx in enumerate(analysis.landmarks_used):
        px, py = pose.pixel(idx)
        col = _YELLOW if i == 1 else _GREEN  # vertex in yellow
        cv2.circle(frame, (px, py), 6, col, -1)


def _draw_info_panel(
    frame: np.ndarray,
    result: ControllerResult,
    *,
    show_rep_count: bool,
    show_stage: bool,
    show_exercise: bool,
) -> None:
    """Draw a semi-transparent info panel in the top-left corner."""
    import cv2

    overlay = frame.copy()
    panel_w, panel_h = 260, 120
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), _BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 40
    rep_state = result.rep_state

    if show_exercise:
        ex_name = _EXERCISE_DISPLAY.get(rep_state.exercise, rep_state.exercise)
        cv2.putText(
            frame, ex_name, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2, cv2.LINE_AA,
        )
        y += 30

    if show_rep_count:
        cv2.putText(
            frame, f"Reps: {rep_state.reps}", (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, _GREEN, 2, cv2.LINE_AA,
        )
        y += 30

    if show_stage:
        stage_col = _GREEN if rep_state.stage == "up" else _RED
        if rep_state.stage == "unknown":
            stage_col = _YELLOW
        cv2.putText(
            frame, f"Stage: {rep_state.stage.upper()}", (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, stage_col, 2, cv2.LINE_AA,
        )
