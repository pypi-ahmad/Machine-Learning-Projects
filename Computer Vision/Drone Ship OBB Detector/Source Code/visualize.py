"""Overlay renderer for Drone Ship OBB Detector.

Draws oriented bounding boxes as quadrilaterals with:
- Class-coloured edges
- Confidence + angle labels
- Per-class count dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OBBConfig
from detector import FrameResult, OBBDetection

# Per-class colour palette (BGR)
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "ship":            (255, 140, 0),
    "large-vehicle":   (0, 200, 255),
    "small-vehicle":   (0, 255, 128),
    "plane":           (255, 80, 80),
    "helicopter":      (200, 0, 200),
    "harbor":          (128, 128, 255),
    "storage-tank":    (180, 180, 50),
    "container-crane": (50, 200, 200),
}
DEFAULT_COLOUR = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame: np.ndarray, result: FrameResult, cfg: OBBConfig) -> np.ndarray:
    """Render oriented boxes and dashboard on *frame* (copy returned)."""
    vis = frame.copy()

    for det in result.detections:
        _draw_obb(vis, det, cfg)

    _draw_dashboard(vis, result)
    return vis


def _draw_obb(vis: np.ndarray, det: OBBDetection, cfg: OBBConfig) -> None:
    """Draw a single oriented bounding box as a quadrilateral."""
    colour = _class_colour(det.class_name)
    pts = det.corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=colour, thickness=cfg.line_width)

    # Label
    parts = [det.class_name]
    if cfg.show_conf:
        parts.append(f"{det.confidence:.0%}")
    if cfg.show_angle:
        parts.append(f"{det.angle_deg:.0f}\u00b0")
    label = " ".join(parts)

    # Position label at topmost corner
    top_idx = int(det.corners[:, 1].argmin())
    lx, ly = int(det.corners[top_idx][0]), int(det.corners[top_idx][1]) - 6

    (tw, th), _ = cv2.getTextSize(label, FONT, cfg.label_font_scale, 1)
    cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly + 2), colour, -1)
    cv2.putText(vis, label, (lx + 2, ly), FONT, cfg.label_font_scale, (255, 255, 255), 1)


def _draw_dashboard(vis: np.ndarray, result: FrameResult) -> None:
    """Semi-transparent count dashboard in top-right corner."""
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22
    num_lines = len(result.class_counts) + 2
    panel_h = line_h * num_lines + margin * 2
    panel_w = 230

    x0, y0 = w - panel_w - margin, margin
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    ty = y0 + margin + line_h
    cv2.putText(vis, "OBB Detections", (x0 + 8, ty), FONT, 0.55, (255, 255, 255), 1)
    ty += line_h

    for cls_name, count in sorted(result.class_counts.items()):
        colour = _class_colour(cls_name)
        cv2.putText(vis, f"{cls_name}: {count}", (x0 + 12, ty), FONT, 0.45, colour, 1)
        ty += line_h

    cv2.putText(vis, f"Total: {result.total}", (x0 + 8, ty), FONT, 0.5, (0, 255, 255), 1)


def _class_colour(name: str) -> tuple[int, int, int]:
    return CLASS_COLOURS.get(name.lower(), DEFAULT_COLOUR)
