"""Cell Counting Instance Segmentation — overlay and visual report rendering."""

from __future__ import annotations

import cv2
import numpy as np

from metrics import CellMetrics
from segmentation import SegmentationResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def draw_cell_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    metrics: CellMetrics,
    cfg=None,
    *,
    alpha: float = 0.35,
    cell_color: tuple[int, int, int] = (0, 220, 100),
    boundary_color: tuple[int, int, int] = (255, 180, 0),
    centroid_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Render cell mask overlay with centroids, bboxes, and stats."""
    if cfg is not None:
        cell_color = cfg.cell_color
        boundary_color = cfg.boundary_color
        centroid_color = cfg.centroid_color
        alpha = cfg.mask_alpha
    canvas = image.copy()

    # Mask overlay
    canvas = _overlay_mask(canvas, seg.combined_mask, cell_color, alpha)

    # Cell contours
    if seg.combined_mask.size > 0:
        contours, _ = cv2.findContours(
            seg.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, boundary_color, 1)

    # Instance centroids + IDs
    for idx, inst in enumerate(seg.instances):
        cx, cy = inst.centroid
        cv2.circle(canvas, (cx, cy), 3, centroid_color, -1)
        label = str(idx + 1)
        cv2.putText(canvas, label, (cx + 4, cy - 4), _FONT, 0.30,
                    _WHITE, 1, cv2.LINE_AA)

    # Stats panel
    lines = [
        f"Cells: {metrics.cell_count}",
        f"Coverage: {metrics.cell_coverage:.1%}",
        f"Mean area: {metrics.mean_cell_area_px:.0f}px",
    ]
    _draw_panel(canvas, lines, x=10, y=24)

    return canvas


def draw_count_badge(
    image: np.ndarray,
    count: int,
) -> np.ndarray:
    """Draw a large cell count badge in the top-right corner."""
    canvas = image.copy()
    label = str(count)
    (tw, th), _ = cv2.getTextSize(label, _FONT, 1.6, 3)
    x = canvas.shape[1] - tw - 20
    y = th + 20
    cv2.rectangle(canvas, (x - 10, 5), (canvas.shape[1] - 5, y + 10), _BLACK, -1)
    cv2.putText(canvas, label, (x, y), _FONT, 1.6, (0, 255, 0), 3, cv2.LINE_AA)
    return canvas


# ── helpers ────────────────────────────────────────────────


def _overlay_mask(image, mask, color, alpha):
    if mask.size == 0:
        return image
    roi = mask > 127
    if not roi.any():
        return image
    out = image.copy()
    overlay = out.copy()
    overlay[roi] = color
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def _draw_panel(canvas, lines, x, y):
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.55, 1)
        cv2.rectangle(canvas, (x - 2, y - th - 4), (x + tw + 4, y + 4), _BLACK, -1)
        cv2.putText(canvas, line, (x, y), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)
        y += th + 12
