"""Visualization module for Face Verification Attendance System.

Draws bounding boxes with identity labels, confidence bars,
unknown markers, and an attendance panel overlay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FaceAttendanceConfig
from parser import AttendanceResult

# ── Colors (BGR) ──────────────────────────────────────────
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_ORANGE = (0, 165, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PANEL_BG = (40, 40, 40)


def draw_overlay(
    frame: np.ndarray,
    result: AttendanceResult,
    cfg: FaceAttendanceConfig,
    *,
    recent_attendance: list[str] | None = None,
) -> np.ndarray:
    """Draw annotated overlay on a frame.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.
    result : AttendanceResult
        Pipeline output.
    cfg : FaceAttendanceConfig
        Display config.
    recent_attendance : list[str], optional
        Recently logged identities for panel display.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    vis = frame.copy()

    # Draw face boxes + labels
    if cfg.show_boxes:
        for m in result.matches:
            x1, y1, x2, y2 = m.box
            color = _GREEN if m.matched else _RED
            lw = cfg.line_width

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, lw)

            if cfg.show_labels:
                label = m.identity
                if cfg.show_confidence:
                    label += f" ({m.similarity:.2f})"
                _draw_label(vis, label, (x1, y1 - 8), color)

            # Confidence bar below box
            if cfg.show_confidence:
                bar_y = y2 + 4
                bar_w = x2 - x1
                fill = int(bar_w * m.similarity)
                cv2.rectangle(vis, (x1, bar_y), (x2, bar_y + 6), _BLACK, -1)
                cv2.rectangle(
                    vis, (x1, bar_y), (x1 + fill, bar_y + 6), color, -1,
                )

    # Summary text
    h, w = vis.shape[:2]
    summary = (
        f"Faces: {result.num_faces} | "
        f"Matched: {result.num_matched} | "
        f"Unknown: {result.num_unknown}"
    )
    cv2.putText(
        vis, summary, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _ORANGE, 2,
    )

    # Attendance panel
    if cfg.show_attendance_panel and recent_attendance:
        _draw_attendance_panel(vis, recent_attendance)

    return vis


def _draw_label(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw text with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = pos
    y = max(y, th + 4)

    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), color, -1)
    cv2.putText(img, text, (x + 2, y), font, scale, _WHITE, thickness)


def _draw_attendance_panel(
    img: np.ndarray,
    names: list[str],
) -> None:
    """Draw a translucent attendance panel in the top-right corner."""
    h, w = img.shape[:2]

    panel_w = 250
    line_h = 28
    panel_h = 40 + len(names) * line_h
    px = w - panel_w - 10
    py = 10

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(
        overlay, (px, py), (px + panel_w, py + panel_h), _PANEL_BG, -1,
    )
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, dst=img)

    # Title
    cv2.putText(
        img, "Attendance Log", (px + 10, py + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2,
    )

    # Names
    for i, name in enumerate(names):
        y = py + 50 + i * line_h
        cv2.putText(
            img, f"  {name}", (px + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1,
        )
