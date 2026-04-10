"""Visualization overlays for Gaze Direction Estimator.

Draws iris markers, eye contours, gaze direction label,
ratio bars, and stats panel onto frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzer import GazeAnalysisResult
from config import GazeConfig
from iris_locator import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR

# Direction → color mapping
_DIR_COLORS = {
    "LEFT": (255, 165, 0),    # orange
    "RIGHT": (255, 165, 0),
    "UP": (200, 100, 255),    # purple
    "DOWN": (200, 100, 255),
    "CENTER": (0, 255, 0),    # green
}


def draw_overlay(
    frame: np.ndarray,
    result: GazeAnalysisResult,
    cfg: GazeConfig,
) -> np.ndarray:
    """Draw gaze analysis overlays on a frame copy.

    Parameters
    ----------
    frame : np.ndarray
        BGR source frame.
    result : GazeAnalysisResult
        Pipeline output for this frame.
    cfg : GazeConfig
        Display settings.

    Returns
    -------
    np.ndarray
        Annotated copy.
    """
    vis = frame.copy()

    if not result.face_detected:
        cv2.putText(
            vis, "No Face Detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        return vis

    lm = result.landmarks

    # Eye contours
    if cfg.show_eye_contours:
        _draw_contour(vis, lm, LEFT_EYE_CONTOUR, (0, 255, 0), cfg.line_width)
        _draw_contour(vis, lm, RIGHT_EYE_CONTOUR, (0, 255, 0), cfg.line_width)

    # Iris center markers
    if cfg.show_iris_markers and result.iris.detected:
        lx, ly = int(result.iris.left_iris_x), int(result.iris.left_iris_y)
        rx, ry = int(result.iris.right_iris_x), int(result.iris.right_iris_y)
        cv2.circle(vis, (lx, ly), 3, (0, 255, 255), -1)
        cv2.circle(vis, (rx, ry), 3, (0, 255, 255), -1)

    # Gaze direction label
    if cfg.show_gaze_label:
        direction = result.direction
        color = _DIR_COLORS.get(direction, (255, 255, 255))
        label = f"Gaze: {direction}"
        conf = result.raw_gaze.confidence
        if conf:
            label += f" ({conf})"
        cv2.putText(
            vis, label, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

    # Ratio display
    if cfg.show_ratios and result.iris.detected:
        y_off = 65
        cv2.putText(
            vis,
            f"H: {result.smoothed.smoothed_h:.2f}  V: {result.smoothed.smoothed_v:.2f}",
            (20, y_off),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )

    # Stats panel
    if cfg.show_stats_panel:
        _draw_stats_panel(vis, result)

    return vis


def _draw_contour(
    img: np.ndarray,
    lm,
    indices: list[int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw polyline around eye landmarks."""
    pts = np.array(
        [lm.pixel_coords(i) for i in indices], dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_stats_panel(img: np.ndarray, result: GazeAnalysisResult) -> None:
    """Draw compact stats panel in the bottom-left."""
    h, _ = img.shape[:2]
    lines = [
        f"Dir: {result.direction}",
        f"H: {result.smoothed.smoothed_h:.2f}",
        f"V: {result.smoothed.smoothed_v:.2f}",
        f"Conf: {result.raw_gaze.confidence}",
    ]
    y = h - 20 * len(lines) - 10
    for line in lines:
        cv2.putText(
            img, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        y += 20
