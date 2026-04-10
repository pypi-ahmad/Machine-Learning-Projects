"""Visualization overlays for Blink Headpose Analyzer.

Draws eye contours, EAR bar, head-pose text, blink counter,
and stats panel onto frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzer import AnalysisResult
from config import AnalyzerConfig
from shared.landmarks import LEFT_EYE, RIGHT_EYE


def draw_overlay(
    frame: np.ndarray,
    result: AnalysisResult,
    cfg: AnalyzerConfig,
) -> np.ndarray:
    """Draw analysis overlays on a frame copy.

    Parameters
    ----------
    frame : np.ndarray
        BGR source frame.
    result : AnalysisResult
        Pipeline output for this frame.
    cfg : AnalyzerConfig
        Display settings.

    Returns
    -------
    np.ndarray
        Annotated copy of the frame.
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
        _draw_eye_contour(vis, lm, LEFT_EYE, (0, 255, 0), cfg.line_width)
        _draw_eye_contour(vis, lm, RIGHT_EYE, (0, 255, 0), cfg.line_width)

    # EAR bar
    if cfg.show_ear_bar:
        _draw_bar(
            vis, "EAR", result.blink.ear,
            cfg.ear_threshold, 0.0, 0.45,
            x=20, y=30, w=200, h=18,
        )

    # Head pose text
    if cfg.show_pose_text:
        pose = result.head_pose
        color = (0, 0, 255) if pose.off_center else (0, 255, 0)
        y_off = 80
        for label, val in [("Yaw", pose.yaw), ("Pitch", pose.pitch), ("Roll", pose.roll)]:
            cv2.putText(
                vis, f"{label}: {val:+.1f}", (20, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
            y_off += 22

    # Stats panel
    if cfg.show_stats_panel:
        _draw_stats_panel(vis, result)

    return vis


def _draw_eye_contour(
    img: np.ndarray,
    lm,
    indices: list[int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a polyline around eye landmarks."""
    pts = np.array(
        [lm.pixel_coords(i) for i in indices], dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_bar(
    img: np.ndarray,
    label: str,
    value: float,
    threshold: float,
    lo: float,
    hi: float,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw a horizontal gauge bar with threshold line."""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    bar_w = int(frac * w)

    color = (0, 255, 0) if value >= threshold else (0, 0, 255)

    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), -1)
    cv2.rectangle(img, (x, y), (x + bar_w, y + h), color, -1)

    # Threshold tick
    t_frac = max(0.0, min(1.0, (threshold - lo) / (hi - lo)))
    t_x = x + int(t_frac * w)
    cv2.line(img, (t_x, y - 3), (t_x, y + h + 3), (255, 255, 255), 1)

    cv2.putText(
        img, f"{label}: {value:.2f}", (x + w + 8, y + h - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )


def _draw_stats_panel(img: np.ndarray, result: AnalysisResult) -> None:
    """Draw a compact stats panel in the bottom-left."""
    h, w = img.shape[:2]
    lines = [
        f"Blinks: {result.blink.total_blinks}",
        f"EAR: {result.blink.ear:.2f}",
        f"Yaw: {result.head_pose.yaw:+.1f}",
        f"Pitch: {result.head_pose.pitch:+.1f}",
    ]
    y = h - 20 * len(lines) - 10
    for line in lines:
        cv2.putText(
            img, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        y += 20
