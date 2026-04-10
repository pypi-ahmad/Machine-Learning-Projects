"""Crop Row & Weed Segmentation — overlay and mask rendering."""

from __future__ import annotations

import cv2
import numpy as np

from class_stats import AreaReport
from config import CropWeedConfig
from segmentation import SegmentationResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def draw_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    report: AreaReport,
    cfg: CropWeedConfig,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    """Render class-coloured mask overlay with labels and stats panel."""
    alpha = alpha if alpha is not None else cfg.mask_alpha
    canvas = image.copy()

    # Draw per-instance masks
    for inst in seg.instances:
        colour = cfg.colour_for(inst.class_name)
        overlay = canvas.copy()
        overlay[inst.mask > 127] = colour
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # Draw bboxes and labels
    for inst in seg.instances:
        colour = cfg.colour_for(inst.class_name)
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = f"{inst.class_name} {inst.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)

    # Stats panel (top-left)
    _draw_stats_panel(canvas, report, cfg)
    # Legend (top-right)
    _draw_legend(canvas, report, cfg)

    return canvas


def render_class_masks(
    seg: SegmentationResult,
    cfg: CropWeedConfig,
) -> np.ndarray:
    """Render an RGB image where each pixel is coloured by class.

    Pixels not covered by any class are black (background/soil).
    """
    h, w = seg.image_hw
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for name, mask in seg.class_masks.items():
        colour = cfg.colour_for(name)
        canvas[mask > 127] = colour
    return canvas


def draw_comparison(
    image: np.ndarray,
    overlay: np.ndarray,
    class_mask_vis: np.ndarray,
) -> np.ndarray:
    """Create a 3-panel horizontal comparison: original | overlay | class map."""
    h = image.shape[0]
    panels = []
    for panel in [image, overlay, class_mask_vis]:
        if panel.shape[:2] != (h, image.shape[1]):
            panel = cv2.resize(panel, (image.shape[1], h))
        panels.append(panel)
    return np.hstack(panels)


# ── internal helpers ───────────────────────────────────────


def _draw_stats_panel(
    canvas: np.ndarray,
    report: AreaReport,
    cfg: CropWeedConfig,
) -> None:
    """Render a stats panel in the top-left corner."""
    lines = [f"Instances: {report.total_instances}"]
    for name, stats in report.per_class.items():
        lines.append(
            f"  {name}: {stats.instance_count} ({stats.coverage_ratio:.1%})"
        )
    lines.append(f"Background: {report.background_ratio:.1%}")

    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.50, 1)
        cv2.rectangle(canvas, (8, y - th - 4), (16 + tw, y + 4), _BLACK, -1)
        cv2.putText(canvas, line, (12, y), _FONT, 0.50, _WHITE, 1, cv2.LINE_AA)
        y += th + 10


def _draw_legend(
    canvas: np.ndarray,
    report: AreaReport,
    cfg: CropWeedConfig,
) -> None:
    """Draw a colour legend in the top-right corner."""
    h, w = canvas.shape[:2]
    names = list(report.per_class.keys()) or cfg.class_names
    bx = w - 150
    by = 10
    for i, name in enumerate(names):
        colour = cfg.colour_for(name)
        y = by + i * 24
        cv2.rectangle(canvas, (bx, y), (bx + 16, y + 16), colour, -1)
        cv2.putText(canvas, name, (bx + 22, y + 13), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)
