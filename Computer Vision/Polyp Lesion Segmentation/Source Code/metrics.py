"""Polyp Lesion Segmentation — per-image polyp metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segmentation import SegmentationResult


@dataclass
class PolypMetrics:
    """Metrics computed from a single image's segmentation result."""

    polyp_area_px: int
    total_image_px: int
    polyp_coverage: float       # polyp / total
    polyp_count: int
    mean_confidence: float
    largest_polyp_px: int       # area of the largest instance

    # ── optional ground-truth comparison ──────────────────
    dice: float | None = None
    iou: float | None = None


def compute_polyp_metrics(
    seg: SegmentationResult,
    gt_mask: np.ndarray | None = None,
) -> PolypMetrics:
    """Derive polyp metrics from a segmentation result.

    Parameters
    ----------
    seg : SegmentationResult
        Model segmentation output.
    gt_mask : np.ndarray | None
        Optional ground-truth binary mask (H, W; 0/255).
        When provided, Dice and IoU are computed.
    """
    h, w = seg.image_hw
    total = h * w if h > 0 and w > 0 else 1

    confs = [inst.confidence for inst in seg.instances]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    largest = max((inst.area_px for inst in seg.instances), default=0)

    dice_val = None
    iou_val = None
    if gt_mask is not None and seg.combined_mask.size > 0:
        dice_val, iou_val = _dice_iou(seg.combined_mask, gt_mask)

    return PolypMetrics(
        polyp_area_px=seg.total_area_px,
        total_image_px=total,
        polyp_coverage=round(seg.total_area_px / total, 6),
        polyp_count=seg.count,
        mean_confidence=round(mean_conf, 4),
        largest_polyp_px=largest,
        dice=dice_val,
        iou=iou_val,
    )


def _dice_iou(
    pred: np.ndarray, gt: np.ndarray,
) -> tuple[float, float]:
    """Compute Dice coefficient and IoU between two binary masks."""
    p = (pred > 127).astype(np.uint8).ravel()
    g = (gt > 127).astype(np.uint8).ravel()

    # Resize if shapes differ
    if p.shape != g.shape:
        import cv2
        gt_resized = cv2.resize(
            gt.astype(np.uint8),
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        g = (gt_resized > 127).astype(np.uint8).ravel()
        p = (pred > 127).astype(np.uint8).ravel()

    intersection = int(np.sum(p & g))
    union = int(np.sum(p | g))
    sum_both = int(np.sum(p)) + int(np.sum(g))

    dice = (2.0 * intersection / sum_both) if sum_both > 0 else 0.0
    iou = (intersection / union) if union > 0 else 0.0
    return round(dice, 6), round(iou, 6)
