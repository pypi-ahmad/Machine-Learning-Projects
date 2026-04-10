"""Waterbody & Flood Extent Segmentation — overlay and visual report rendering."""

from __future__ import annotations

import cv2
import numpy as np

from coverage import CoverageMetrics, FloodChangeMetrics
from flood_compare import ComparisonResult
from segmentation import SegmentationResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


# ── single-image overlay ──────────────────────────────────


def draw_water_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    metrics: CoverageMetrics,
    cfg=None,
    *,
    alpha: float = 0.40,
    water_color: tuple[int, int, int] = (230, 160, 30),
) -> np.ndarray:
    """Render water mask overlay with bounding boxes and stats."""
    if cfg is not None:
        water_color = cfg.water_color
        alpha = cfg.mask_alpha
    canvas = image.copy()

    # Mask overlay
    overlay = canvas.copy()
    overlay[seg.combined_mask > 127] = water_color
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # Instance bboxes
    for inst in seg.instances:
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), water_color, 2)
        label = f"water {inst.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), water_color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)

    # Stats panel
    lines = [
        f"Water: {metrics.coverage_ratio:.1%} ({metrics.water_area_px:,}px)",
        f"Instances: {metrics.instance_count}",
    ]
    _draw_panel(canvas, lines, x=10, y=24)

    return canvas


# ── before / after comparison overlay ──────────────────────


def draw_comparison_overlay(
    after_image: np.ndarray,
    comparison: ComparisonResult,
    *,
    alpha: float = 0.40,
    flood_new_color: tuple[int, int, int] = (0, 0, 220),
    flood_receded_color: tuple[int, int, int] = (0, 200, 0),
    permanent_color: tuple[int, int, int] = (230, 160, 30),
) -> np.ndarray:
    """Overlay flood-change regions on the *after* image."""
    canvas = after_image.copy()

    canvas = _overlay_mask(canvas, comparison.permanent_mask, permanent_color, alpha * 0.5)
    canvas = _overlay_mask(canvas, comparison.flooded_new_mask, flood_new_color, alpha)
    canvas = _overlay_mask(canvas, comparison.receded_mask, flood_receded_color, alpha)

    for region in comparison.regions:
        color = flood_new_color if region.label == "flooded_new" else flood_receded_color
        x, y, w, h = region.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        tag = f"{region.label} ({region.area_px:,}px)"
        cv2.putText(canvas, tag, (x, y - 6), _FONT, 0.40, color, 1, cv2.LINE_AA)

    _draw_flood_legend(canvas, flood_new_color, flood_receded_color, permanent_color)

    return canvas


def draw_side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    *,
    alpha: float = 0.35,
    water_color: tuple[int, int, int] = (230, 160, 30),
) -> np.ndarray:
    """Horizontal concat of before/after with water mask overlays."""
    left = _overlay_mask(before.copy(), before_mask, water_color, alpha)
    right = _overlay_mask(after.copy(), after_mask, water_color, alpha)

    cv2.putText(left, "BEFORE", (10, 30), _FONT, 0.8, _WHITE, 2, cv2.LINE_AA)
    cv2.putText(right, "AFTER", (10, 30), _FONT, 0.8, _WHITE, 2, cv2.LINE_AA)

    return np.hstack([left, right])


def draw_metrics_panel(m: FloodChangeMetrics, width: int = 420) -> np.ndarray:
    """Render a text panel with comparison metrics."""
    lines = [
        f"Before water:     {m.before_water_px:,}px  ({m.before_coverage:.2%})",
        f"After water:      {m.after_water_px:,}px  ({m.after_coverage:.2%})",
        f"New flooding:     {m.flooded_new_px:,}px  ({m.num_new_regions} regions)",
        f"Receded:          {m.receded_px:,}px  ({m.num_receded_regions} regions)",
        f"IoU:              {m.iou:.4f}",
        f"Net change:       {m.net_change_ratio:+.4f}",
    ]
    line_h = 28
    pad = 16
    height = pad * 2 + line_h * len(lines)
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    y = pad + 20
    for line in lines:
        cv2.putText(panel, line, (pad, y), _FONT, 0.50, _WHITE, 1, cv2.LINE_AA)
        y += line_h
    return panel


def compose_comparison_report(
    before: np.ndarray,
    after: np.ndarray,
    before_seg: SegmentationResult,
    after_seg: SegmentationResult,
    comparison: ComparisonResult,
    metrics: FloodChangeMetrics,
    cfg=None,
    *,
    alpha: float = 0.40,
) -> np.ndarray:
    """Full visual report: side-by-side + change overlay + metrics."""
    if cfg is not None:
        alpha = cfg.mask_alpha
    before_mask = before_seg.combined_mask
    after_mask = after_seg.combined_mask
    sbs = draw_side_by_side(before, after, before_mask, after_mask, alpha=alpha)
    total_w = sbs.shape[1]

    change = draw_comparison_overlay(after, comparison, alpha=alpha)
    change_resized = cv2.resize(
        change, (total_w, int(change.shape[0] * total_w / change.shape[1])),
        interpolation=cv2.INTER_LINEAR,
    )
    metrics_img = draw_metrics_panel(metrics, width=total_w)

    return np.vstack([sbs, change_resized, metrics_img])


# ── helpers ────────────────────────────────────────────────


def _overlay_mask(image, mask, color, alpha):
    out = image.copy()
    roi = mask > 127
    if not roi.any():
        return out
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


def _draw_flood_legend(canvas, new_c, rec_c, perm_c):
    h, w = canvas.shape[:2]
    items = [(new_c, "New flooding"), (rec_c, "Receded"), (perm_c, "Permanent")]
    bx = w - 170
    by = 10
    for i, (color, label) in enumerate(items):
        y = by + i * 24
        cv2.rectangle(canvas, (bx, y), (bx + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (bx + 22, y + 13), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)
