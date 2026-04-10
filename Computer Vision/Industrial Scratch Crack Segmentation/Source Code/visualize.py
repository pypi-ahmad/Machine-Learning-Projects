"""Industrial Scratch / Crack Segmentation — overlay and visual report."""

from __future__ import annotations

import cv2
import numpy as np

from metrics import DefectMetrics
from segmentation import SegmentationResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

_SEVERITY_COLORS = {
    "none": (0, 200, 0),       # green
    "low": (0, 220, 220),      # yellow
    "medium": (0, 140, 255),   # orange
    "high": (0, 0, 255),       # red
}


def draw_defect_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    metrics: DefectMetrics,
    cfg=None,
    *,
    alpha: float = 0.40,
    defect_color: tuple[int, int, int] = (0, 0, 255),
    boundary_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Render defect mask overlay with bboxes, lengths, and stats panel."""
    if cfg is not None:
        defect_color = cfg.scratch_color
        boundary_color = cfg.boundary_color
        alpha = cfg.mask_alpha
    canvas = image.copy()

    # Mask overlay
    canvas = _overlay_mask(canvas, seg.combined_mask, defect_color, alpha)

    # Defect contours
    if seg.combined_mask.size > 0:
        contours, _ = cv2.findContours(
            seg.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, boundary_color, 1)

    # Instance bboxes + length label
    for inst in seg.instances:
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), defect_color, 1)
        label = f"{inst.length_px:.0f}px {inst.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.35, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 4, y1), defect_color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 2), _FONT, 0.35, _WHITE, 1,
                    cv2.LINE_AA)

    # Stats + severity panel
    sev_color = _SEVERITY_COLORS.get(metrics.severity, _WHITE)
    lines = [
        f"Defects: {metrics.defect_count}",
        f"Coverage: {metrics.defect_coverage:.2%}",
        f"Severity: {metrics.severity.upper()}",
        f"Max length: {metrics.max_length_px:.0f}px",
    ]
    _draw_panel(canvas, lines, x=10, y=24, highlight_line=2,
                highlight_color=sev_color)

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


def _draw_panel(canvas, lines, x, y, highlight_line=-1, highlight_color=_WHITE):
    for i, line in enumerate(lines):
        color = highlight_color if i == highlight_line else _WHITE
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.55, 1)
        cv2.rectangle(canvas, (x - 2, y - th - 4), (x + tw + 4, y + 4), _BLACK, -1)
        cv2.putText(canvas, line, (x, y), _FONT, 0.55, color, 1, cv2.LINE_AA)
        y += th + 12
