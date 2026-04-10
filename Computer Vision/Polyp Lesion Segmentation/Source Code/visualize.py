"""Polyp Lesion Segmentation — overlay and visual report rendering."""

from __future__ import annotations

import cv2
import numpy as np

from metrics import PolypMetrics
from segmentation import SegmentationResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def draw_polyp_overlay(
    image: np.ndarray,
    seg: SegmentationResult,
    metrics: PolypMetrics,
    cfg=None,
    *,
    alpha: float = 0.40,
    polyp_color: tuple[int, int, int] = (0, 100, 255),
    boundary_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Render polyp mask overlay with bounding boxes and stats."""
    if cfg is not None:
        polyp_color = cfg.polyp_color
        boundary_color = cfg.boundary_color
        alpha = cfg.mask_alpha
    canvas = image.copy()

    # Mask overlay
    canvas = _overlay_mask(canvas, seg.combined_mask, polyp_color, alpha)

    # Polyp contours
    if seg.combined_mask.size > 0:
        contours, _ = cv2.findContours(
            seg.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, boundary_color, 2)

    # Instance bboxes
    for inst in seg.instances:
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), polyp_color, 2)
        label = f"polyp {inst.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), polyp_color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), _FONT, 0.45, _WHITE, 1,
                    cv2.LINE_AA)

    # Stats panel
    lines = [
        f"Polyp: {metrics.polyp_coverage:.1%} ({metrics.polyp_area_px:,}px)",
        f"Regions: {metrics.polyp_count}",
    ]
    if metrics.dice is not None:
        lines.append(f"Dice: {metrics.dice:.4f}  IoU: {metrics.iou:.4f}")
    _draw_panel(canvas, lines, x=10, y=24)

    return canvas


def draw_gt_comparison(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    metrics: PolypMetrics,
    cfg=None,
) -> np.ndarray:
    """Side-by-side comparison: prediction vs ground truth."""
    h, w = image.shape[:2]

    # Prediction overlay
    pred_vis = image.copy()
    pred_vis = _overlay_mask(pred_vis, pred_mask, (0, 100, 255), 0.40)

    # GT overlay
    gt_vis = image.copy()
    gt_vis = _overlay_mask(gt_vis, gt_mask, (0, 255, 0), 0.40)

    # Labels
    cv2.putText(pred_vis, "Prediction", (10, 30), _FONT, 0.7, _WHITE, 2, cv2.LINE_AA)
    cv2.putText(gt_vis, "Ground Truth", (10, 30), _FONT, 0.7, _WHITE, 2, cv2.LINE_AA)

    combined = np.hstack([pred_vis, gt_vis])

    # Metrics bar
    if metrics.dice is not None:
        bar_h = 36
        bar = np.full((bar_h, combined.shape[1], 3), 30, dtype=np.uint8)
        text = (f"Dice: {metrics.dice:.4f}  |  IoU: {metrics.iou:.4f}  |  "
                f"Coverage: {metrics.polyp_coverage:.2%}")
        cv2.putText(bar, text, (10, 24), _FONT, 0.50, _WHITE, 1, cv2.LINE_AA)
        combined = np.vstack([combined, bar])

    return combined


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
