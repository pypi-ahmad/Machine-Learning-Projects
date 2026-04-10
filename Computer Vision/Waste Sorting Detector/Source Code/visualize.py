"""Overlay renderer for Waste Sorting Detector.

Draws:
- Bounding boxes coloured by waste class
- Bin-zone polygons (transparent overlay)
- Per-class count dashboard
- Misplacement alerts (red highlight)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WasteConfig
from sorter import FrameResult, WasteItem

# Per-class colour palette — distinct colours for each waste type
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "plastic":   (255, 165, 0),    # orange
    "paper":     (0, 200, 255),    # cyan
    "cardboard": (50, 120, 200),   # brown-ish
    "metal":     (180, 180, 180),  # grey
    "glass":     (0, 255, 128),    # green
    "trash":     (80, 80, 80),     # dark grey
}
DEFAULT_COLOUR = (200, 200, 0)
ZONE_ALPHA = 0.25
MISPLACED_COLOUR = (0, 0, 255)  # red
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame: np.ndarray, result: FrameResult, cfg: WasteConfig) -> np.ndarray:
    """Render all visual elements on *frame* (mutated in-place) and return it."""
    vis = frame.copy()

    # 1. Bin-zone polygons
    if cfg.bin_zones:
        _draw_zones(vis, cfg)

    # 2. Detection boxes
    for item in result.items:
        _draw_box(vis, item, cfg)

    # 3. Dashboard
    if cfg.show_counts:
        _draw_dashboard(vis, result)

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _draw_zones(vis: np.ndarray, cfg: WasteConfig) -> None:
    overlay = vis.copy()
    for zone in cfg.bin_zones:
        pts = np.array(zone.polygon, dtype=np.int32)
        colour = _zone_colour(zone.name)
        cv2.fillPoly(overlay, [pts], colour)
        cv2.polylines(vis, [pts], isClosed=True, color=colour, thickness=2)
        # Label
        cx = int(np.mean([p[0] for p in zone.polygon]))
        cy = int(np.mean([p[1] for p in zone.polygon]))
        cv2.putText(vis, zone.name, (cx - 40, cy), FONT, 0.6, colour, 2)
    cv2.addWeighted(overlay, ZONE_ALPHA, vis, 1 - ZONE_ALPHA, 0, vis)


def _draw_box(vis: np.ndarray, item: WasteItem, cfg: WasteConfig) -> None:
    x1, y1, x2, y2 = item.bbox
    colour = MISPLACED_COLOUR if item.misplaced else _class_colour(item.class_name)
    thickness = 3 if item.misplaced else 2
    cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)

    label = f"{item.class_name} {item.confidence:.0%}"
    if item.misplaced:
        label += " MISPLACED"
    elif item.zone_name:
        label += f" [{item.zone_name}]"

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(vis, label, (x1 + 2, y1 - 4), FONT, 0.5, (255, 255, 255), 1)


def _draw_dashboard(vis: np.ndarray, result: FrameResult) -> None:
    h, w = vis.shape[:2]
    margin = 10
    line_h = 22
    num_lines = len(result.class_counts) + 2  # header + total
    panel_h = line_h * num_lines + margin * 2
    panel_w = 220

    x0, y0 = w - panel_w - margin, margin
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    ty = y0 + margin + line_h
    cv2.putText(vis, "Waste Counts", (x0 + 8, ty), FONT, 0.55, (255, 255, 255), 1)
    ty += line_h

    for cls_name, count in sorted(result.class_counts.items()):
        colour = _class_colour(cls_name)
        cv2.putText(vis, f"{cls_name}: {count}", (x0 + 12, ty), FONT, 0.45, colour, 1)
        ty += line_h

    cv2.putText(vis, f"Total: {result.total_items}", (x0 + 8, ty), FONT, 0.5, (0, 255, 255), 1)

    # Misplaced count
    if result.misplaced_items:
        ty += line_h
        cv2.putText(
            vis,
            f"Misplaced: {len(result.misplaced_items)}",
            (x0 + 8, ty),
            FONT, 0.5, MISPLACED_COLOUR, 1,
        )


def _class_colour(name: str) -> tuple[int, int, int]:
    return CLASS_COLOURS.get(name.lower(), DEFAULT_COLOUR)


def _zone_colour(name: str) -> tuple[int, int, int]:
    """Deterministic colour per zone name."""
    h = hash(name) % 360
    hsv = np.uint8([[[h // 2, 180, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
