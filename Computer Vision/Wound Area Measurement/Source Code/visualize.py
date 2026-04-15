"""Wound Area Measurement -- overlay and visual report rendering."""

from __future__ import annotations

import cv2
import numpy as np

from change_tracker import ChangeSummary
from metrics import WoundMetrics
from segmentation import SegmentationResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def draw_wound_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    metrics: WoundMetrics,
    cfg=None,
    *,
    alpha: float = 0.40,
    wound_color: tuple[int, int, int] = (0, 0, 220),
    boundary_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Render wound mask overlay with bounding boxes and stats."""
    if cfg is not None:
        wound_color = cfg.wound_color
        boundary_color = cfg.boundary_color
        alpha = cfg.mask_alpha
    canvas = image.copy()

    # Mask overlay
    canvas = _overlay_mask(canvas, seg.combined_mask, wound_color, alpha)

    # Wound contours
    if seg.combined_mask.size > 0:
        contours, _ = cv2.findContours(
            seg.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, boundary_color, 2)

    # Instance bboxes
    for inst in seg.instances:
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), wound_color, 2)
        label = f"wound {inst.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), wound_color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), _FONT, 0.45, _WHITE, 1,
                    cv2.LINE_AA)

    # Stats panel
    lines = [
        f"Wound: {metrics.wound_coverage:.1%} ({metrics.wound_area_px:,}px)",
        f"Regions: {metrics.wound_count}",
    ]
    _draw_panel(canvas, lines, x=10, y=24)

    return canvas


def draw_change_panel(summary: ChangeSummary, width: int = 400) -> np.ndarray:
    """Render a text panel summarising wound area changes over a series."""
    lines = [
        f"Series: {summary.total_images} image(s)",
        f"Initial area: {summary.initial_area_px:,}px",
        f"Final area:   {summary.final_area_px:,}px",
        f"Net change:   {summary.net_change_px:+,}px ({summary.net_change_ratio:+.2%})",
        f"Peak area:    {summary.peak_area_px:,}px",
    ]
    line_h = 26
    pad = 12
    height = pad * 2 + line_h * len(lines)
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    y = pad + 18
    for line in lines:
        cv2.putText(panel, line, (pad, y), _FONT, 0.48, _WHITE, 1, cv2.LINE_AA)
        y += line_h
    return panel


def compose_report(
    image: np.ndarray,
    seg: SegmentationResult,
    metrics: WoundMetrics,
    summary: ChangeSummary | None = None,
    cfg=None,
) -> np.ndarray:
    """Full visual report: wound overlay + optional change summary panel."""
    overlay = draw_wound_overlay(image, seg, metrics, cfg)
    if summary is None or summary.total_images <= 1:
        return overlay
    panel = draw_change_panel(summary, width=overlay.shape[1])
    return np.vstack([overlay, panel])


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
