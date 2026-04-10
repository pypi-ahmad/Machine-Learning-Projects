"""Overlay renderer for Number Plate Reader Pro.

Draws detection bounding boxes, plate text labels, confidence
scores, and a summary panel on video frames or images.

Usage::

    from visualize import draw_overlay

    vis = draw_overlay(frame, plate_read_result, cfg)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PlateConfig
from parser import PlateReadResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_BG = (30, 30, 30)
NEW_COLOUR = (0, 255, 0)
DUP_COLOUR = (128, 128, 128)
INVALID_COLOUR = (0, 0, 220)
TEXT_COLOUR = (255, 255, 255)


def draw_overlay(
    image: np.ndarray,
    result: PlateReadResult,
    cfg: PlateConfig,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    if cfg.show_det_boxes:
        _draw_detections(vis, result, cfg)

    _draw_panel(vis, result)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _draw_detections(
    vis: np.ndarray,
    result: PlateReadResult,
    cfg: PlateConfig,
) -> None:
    """Draw bounding boxes and plate text labels."""
    for read in result.reads:
        x1, y1, x2, y2 = read.box

        # Colour: green for new, grey for duplicate, red for invalid
        if not read.is_valid:
            colour = INVALID_COLOUR
        elif read.is_new:
            colour = NEW_COLOUR
        else:
            colour = DUP_COLOUR

        # Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, cfg.line_width)

        # Label
        parts: list[str] = []
        if cfg.show_plate_text:
            parts.append(read.plate_text or "???")
        if cfg.show_confidence:
            parts.append(f"D:{read.det_confidence:.2f}")
            parts.append(f"O:{read.ocr_confidence:.2f}")
        if not read.is_new:
            parts.append("[dup]")

        label = " | ".join(parts)
        if label:
            (tw, th), baseline = cv2.getTextSize(label, FONT, 0.50, 1)
            # Background rectangle
            cv2.rectangle(
                vis,
                (x1, y1 - th - baseline - 6),
                (x1 + tw + 4, y1),
                PANEL_BG,
                -1,
            )
            cv2.putText(
                vis, label, (x1 + 2, y1 - baseline - 3),
                FONT, 0.50, colour, 1,
            )


def _draw_panel(vis: np.ndarray, result: PlateReadResult) -> None:
    """Draw a summary panel in the top-left corner."""
    h, w = vis.shape[:2]
    margin = 8
    line_h = 20

    lines = [
        f"Frame: {result.frame_index}",
        f"Detected: {result.num_detections}",
        f"Valid: {result.num_valid}",
        f"New: {result.num_new}",
    ]

    # Add plate texts
    for read in result.reads:
        if read.is_new and read.plate_text:
            lines.append(f"  > {read.plate_text}")

    panel_h = margin * 2 + line_h * len(lines)
    panel_w = 260

    # Semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.70, vis, 0.30, 0, vis)

    for i, line in enumerate(lines):
        y = margin + (i + 1) * line_h - 4
        cv2.putText(vis, line, (margin, y), FONT, 0.45, TEXT_COLOUR, 1)
