"""Fire Area Segmentation — overlay and visual report rendering."""

from __future__ import annotations

import cv2
import numpy as np

from metrics import FrameMetrics
from segmentation import SegmentationResult
from trend import TrendSummary

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def draw_fire_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    metrics: FrameMetrics,
    cfg=None,
    *,
    alpha: float = 0.45,
    fire_color: tuple[int, int, int] = (0, 60, 255),
    smoke_color: tuple[int, int, int] = (180, 180, 180),
) -> np.ndarray:
    """Render fire/smoke mask overlay with bounding boxes and stats."""
    if cfg is not None:
        fire_color = cfg.fire_color
        smoke_color = cfg.smoke_color
        alpha = cfg.mask_alpha
    canvas = image.copy()

    # Mask overlays
    canvas = _overlay_mask(canvas, seg.fire_mask, fire_color, alpha)
    canvas = _overlay_mask(canvas, seg.smoke_mask, smoke_color, alpha * 0.7)

    # Instance bboxes
    for inst in seg.instances:
        color = fire_color if inst.class_name == "fire" else smoke_color
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{inst.class_name} {inst.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), _FONT, 0.45, _WHITE, 1,
                    cv2.LINE_AA)

    # Stats panel
    lines = [
        f"Fire: {metrics.fire_coverage:.1%} ({metrics.fire_count} rgn)",
        f"Smoke: {metrics.smoke_coverage:.1%} ({metrics.smoke_count} rgn)",
    ]
    _draw_panel(canvas, lines, x=10, y=24)

    # Legend
    _draw_legend(canvas, fire_color, smoke_color)

    return canvas


def draw_trend_panel(trend: TrendSummary, width: int = 380) -> np.ndarray:
    """Render a text panel summarising the rolling trend."""
    lines = [
        f"Window: {trend.frames_seen}/{trend.window_size} frames",
        f"Avg fire coverage:  {trend.avg_fire_coverage:.2%}",
        f"Peak fire coverage: {trend.peak_fire_coverage:.2%}",
        f"Fire growth rate:   {trend.fire_growth_rate:+.4%}/frame",
        f"Avg smoke coverage: {trend.avg_smoke_coverage:.2%}",
        f"Smoke growth rate:  {trend.smoke_growth_rate:+.4%}/frame",
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
    metrics: FrameMetrics,
    trend: TrendSummary | None,
    cfg=None,
) -> np.ndarray:
    """Full frame: overlay + optional trend panel below."""
    overlay = draw_fire_overlay(image, seg, metrics, cfg)
    if trend is None:
        return overlay
    panel = draw_trend_panel(trend, width=overlay.shape[1])
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


def _draw_legend(canvas, fire_c, smoke_c):
    h, w = canvas.shape[:2]
    items = [(fire_c, "Fire"), (smoke_c, "Smoke")]
    bx = w - 120
    by = 10
    for i, (color, label) in enumerate(items):
        y = by + i * 24
        cv2.rectangle(canvas, (bx, y), (bx + 16, y + 16), color, -1)
        cv2.putText(canvas, label, (bx + 22, y + 13), _FONT, 0.45, _WHITE, 1,
                    cv2.LINE_AA)
