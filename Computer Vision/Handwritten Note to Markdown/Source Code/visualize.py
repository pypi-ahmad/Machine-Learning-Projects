"""Overlay renderer for Handwritten Note to Markdown.

Draws line-segmentation boxes with per-line confidence scores
and recognised text on the source image.

Usage::

    from visualize import draw_overlay

    vis = draw_overlay(image, parse_result, cfg)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import NoteConfig
from parser import NoteParseResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_BG = (30, 30, 30)

# Colour ramp: red (low) → yellow → green (high)
def _confidence_colour(conf: float) -> tuple[int, int, int]:
    """Map confidence [0,1] to BGR colour."""
    if conf >= 0.75:
        return (0, 200, 0)       # green
    if conf >= 0.50:
        return (0, 200, 200)     # yellow
    return (0, 0, 220)           # red


def draw_overlay(
    image: np.ndarray,
    result: NoteParseResult,
    cfg: NoteConfig,
) -> np.ndarray:
    """Render annotated overlay on *image* (copy returned)."""
    vis = image.copy()

    if cfg.show_line_boxes:
        _draw_line_boxes(vis, result, cfg)

    _draw_panel(vis, result, cfg)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _draw_line_boxes(
    vis: np.ndarray,
    result: NoteParseResult,
    cfg: NoteConfig,
) -> None:
    h, w = vis.shape[:2]

    for line in result.lines:
        text = line.text.strip()
        if not text:
            continue

        colour = _confidence_colour(line.confidence)

        # Line bounding box (full width)
        y0 = max(0, line.y_start - cfg.line_width)
        y1 = min(h, line.y_end + cfg.line_width)
        cv2.rectangle(vis, (0, y0), (w, y1), colour, cfg.line_width)

        # Label: recognised text (truncated) + confidence
        label = text[:50]
        if cfg.show_confidence:
            label = f"[{line.confidence:.2f}] {label}"

        (tw, th), _ = cv2.getTextSize(label, FONT, 0.40, 1)
        label_y = y0 - 6 if y0 > 20 else y1 + th + 6
        cv2.rectangle(
            vis, (0, label_y - th - 3), (tw + 8, label_y + 4),
            PANEL_BG, -1,
        )
        cv2.putText(vis, label, (4, label_y), FONT, 0.40, colour, 1)


def _draw_panel(
    vis: np.ndarray,
    result: NoteParseResult,
    cfg: NoteConfig,
) -> None:
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22

    nonempty = sum(1 for ln in result.lines if ln.text.strip())

    lines_text: list[tuple[str, tuple[int, int, int]]] = [
        ("Handwritten OCR", (255, 255, 255)),
        ("-" * 28, (120, 120, 120)),
        (f"Lines: {nonempty}/{result.num_lines}", (180, 180, 180)),
        (
            f"Mean conf: {result.mean_confidence:.2f}",
            _confidence_colour(result.mean_confidence),
        ),
    ]

    # Show first 10 lines in panel
    for ln in result.lines[:10]:
        text = ln.text.strip()
        if not text:
            continue
        display = text if len(text) <= 35 else text[:32] + "..."
        colour = _confidence_colour(ln.confidence)
        lines_text.append((display, colour))

    if result.num_lines > 10:
        lines_text.append(
            (f"  ... +{result.num_lines - 10} more", (140, 140, 140)),
        )

    panel_h = line_h * len(lines_text) + margin * 2
    panel_w = 380
    x0 = w - panel_w - margin
    y0 = margin

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    ty = y0 + margin + line_h
    for text, colour in lines_text:
        cv2.putText(vis, text, (x0 + 8, ty), FONT, 0.38, colour, 1)
        ty += line_h
