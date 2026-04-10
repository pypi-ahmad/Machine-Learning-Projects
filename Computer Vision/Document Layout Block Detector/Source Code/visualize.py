"""Overlay renderer for Document Layout Block Detector.

Draws:
- Coloured bounding boxes per layout class
- Class + confidence labels
- Block ID annotations
- Per-class count dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LayoutConfig
from detector import LayoutBlock, PageResult

# Per-class colour palette (BGR)
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "title":       (0, 100, 255),    # orange-red
    "text":        (200, 180, 50),   # teal
    "table":       (0, 200, 0),      # green
    "figure":      (255, 100, 100),  # blue
    "list":        (0, 200, 200),    # yellow
    "caption":     (180, 100, 255),  # pink
    "header":      (255, 200, 0),    # cyan
    "footer":      (128, 128, 128),  # grey
    "page-number": (100, 100, 100),  # dark grey
    "stamp":       (0, 0, 255),      # red
}
DEFAULT_COLOUR = (180, 180, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(image: np.ndarray, result: PageResult, cfg: LayoutConfig) -> np.ndarray:
    """Render layout blocks and dashboard on *image* (copy returned)."""
    vis = image.copy()

    for block in result.blocks:
        _draw_block(vis, block, cfg)

    _draw_dashboard(vis, result)
    return vis


def _draw_block(vis: np.ndarray, block: LayoutBlock, cfg: LayoutConfig) -> None:
    x1, y1, x2, y2 = block.bbox
    colour = _class_colour(block.class_name)

    # Semi-transparent fill
    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)

    # Border
    cv2.rectangle(vis, (x1, y1), (x2, y2), colour, cfg.line_width)

    # Label
    parts = [f"#{block.block_id}", block.class_name]
    if cfg.show_conf:
        parts.append(f"{block.confidence:.0%}")
    label = " ".join(parts)

    (tw, th), _ = cv2.getTextSize(label, FONT, cfg.label_font_scale, 1)
    cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(vis, label, (x1 + 2, y1 - 4), FONT, cfg.label_font_scale, (255, 255, 255), 1)


def _draw_dashboard(vis: np.ndarray, result: PageResult) -> None:
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22
    num_lines = len(result.class_counts) + 2
    panel_h = line_h * num_lines + margin * 2
    panel_w = 220

    x0, y0 = w - panel_w - margin, margin
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    ty = y0 + margin + line_h
    cv2.putText(vis, "Layout Blocks", (x0 + 8, ty), FONT, 0.55, (255, 255, 255), 1)
    ty += line_h

    for cls_name, count in sorted(result.class_counts.items()):
        colour = _class_colour(cls_name)
        cv2.putText(vis, f"{cls_name}: {count}", (x0 + 12, ty), FONT, 0.45, colour, 1)
        ty += line_h

    cv2.putText(vis, f"Total: {result.total_blocks}", (x0 + 8, ty), FONT, 0.5, (0, 255, 255), 1)


def _class_colour(name: str) -> tuple[int, int, int]:
    return CLASS_COLOURS.get(name.lower(), DEFAULT_COLOUR)
