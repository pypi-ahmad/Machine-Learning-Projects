"""Visualization module for Driver Drowsiness Monitor.

Draws eye/mouth contours, EAR/MAR bars, head pose indicators,
alert banners, and a stats panel overlay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DrowsinessConfig
from landmark_detector import LEFT_EYE, MOUTH_HORIZONTAL, RIGHT_EYE
from parser import DrowsinessResult

# ── Colors (BGR) ──────────────────────────────────────────
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)
_ORANGE = (0, 165, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PANEL_BG = (40, 40, 40)
_ALERT_BG = (0, 0, 180)


def draw_overlay(
    frame: np.ndarray,
    result: DrowsinessResult,
    cfg: DrowsinessConfig,
) -> np.ndarray:
    """Draw annotated overlay on a frame.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.
    result : DrowsinessResult
        Pipeline output.
    cfg : DrowsinessConfig
        Display config.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    if not result.face_detected:
        cv2.putText(
            vis, "No face detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2,
        )
        return vis

    lm = result.landmarks

    # Eye contours
    if cfg.show_eye_contours:
        for eye_idx in [LEFT_EYE, RIGHT_EYE]:
            pts = [
                tuple(int(c) for c in lm.pixel_coords(i))
                for i in eye_idx
            ]
            color = _RED if result.blink.eyes_closed else _GREEN
            for i in range(len(pts)):
                cv2.line(vis, pts[i], pts[(i + 1) % len(pts)], color, cfg.line_width)

    # Mouth contours
    if cfg.show_mouth_contours:
        mouth_pts = [
            tuple(int(c) for c in lm.pixel_coords(i))
            for i in MOUTH_HORIZONTAL
        ]
        color = _YELLOW if result.yawn.mouth_open else _GREEN
        if len(mouth_pts) >= 2:
            cv2.line(vis, mouth_pts[0], mouth_pts[1], color, cfg.line_width)

    # EAR bar
    if cfg.show_ear_bar:
        _draw_bar(
            vis, 10, h - 60, 150, 20,
            result.blink.ear, max_val=0.4,
            label=f"EAR: {result.blink.ear:.2f}",
            threshold=cfg.ear_threshold,
            invert=True,  # lower is worse for EAR
        )

    # MAR bar
    if cfg.show_mar_bar:
        _draw_bar(
            vis, 10, h - 30, 150, 20,
            result.yawn.mar, max_val=1.0,
            label=f"MAR: {result.yawn.mar:.2f}",
            threshold=cfg.mar_threshold,
            invert=False,
        )

    # Head pose text
    pose = result.head_pose
    pose_color = _RED if pose.off_center else _GREEN
    cv2.putText(
        vis,
        f"Yaw: {pose.yaw:.0f}  Pitch: {pose.pitch:.0f}",
        (w - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1,
    )

    # Stats panel
    if cfg.show_stats_panel:
        _draw_stats_panel(vis, result, cfg)

    # Alert banners
    if cfg.show_alerts and result.active_alerts:
        _draw_alert_banner(vis, result.active_alerts)

    return vis


def _draw_bar(
    img: np.ndarray,
    x: int, y: int, bar_w: int, bar_h: int,
    value: float, max_val: float,
    label: str,
    threshold: float,
    invert: bool = False,
) -> None:
    """Draw a horizontal metric bar with threshold line."""
    # Background
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), _BLACK, -1)

    # Fill
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    fill_w = int(bar_w * ratio)

    if invert:
        color = _RED if value < threshold else _GREEN
    else:
        color = _RED if value > threshold else _GREEN

    cv2.rectangle(img, (x, y), (x + fill_w, y + bar_h), color, -1)

    # Threshold line
    thr_x = int(x + bar_w * min(threshold / max_val, 1.0))
    cv2.line(img, (thr_x, y), (thr_x, y + bar_h), _YELLOW, 1)

    # Label
    cv2.putText(
        img, label, (x + bar_w + 5, y + bar_h - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1,
    )


def _draw_stats_panel(
    img: np.ndarray,
    result: DrowsinessResult,
    cfg: DrowsinessConfig,
) -> None:
    """Draw translucent stats panel in top-left."""
    lines = [
        f"Blinks: {result.blink.total_blinks}",
        f"PERCLOS: {result.blink.perclos:.1%}",
        f"Yawns: {result.yawn.total_yawns}",
        f"Alerts: {len(result.alerts)}",
    ]

    panel_w = 180
    line_h = 22
    panel_h = 30 + len(lines) * line_h
    px, py = 10, 10

    overlay = img.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, dst=img)

    cv2.putText(
        img, "Driver Monitor", (px + 8, py + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 1,
    )
    for i, line in enumerate(lines):
        y = py + 40 + i * line_h
        cv2.putText(
            img, line, (px + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1,
        )


def _draw_alert_banner(
    img: np.ndarray,
    active: set[str],
) -> None:
    """Draw red alert banner at the top of the frame."""
    h, w = img.shape[:2]
    banner_h = 40

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _ALERT_BG, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)

    text = "ALERT: " + " | ".join(sorted(active)).upper()
    cv2.putText(
        img, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2,
    )
