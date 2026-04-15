"""Waterbody & Flood Extent Segmentation — coverage metrics.
"""Waterbody & Flood Extent Segmentation — coverage metrics.

Computes water coverage for single images and changed-area metrics for
before/after comparison mode.
"""
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flood_compare import ComparisonResult
from segmentation import SegmentationResult


# ── single-image metrics ──────────────────────────────────


@dataclass
class CoverageMetrics:
    """Water coverage metrics for a single image."""

    water_area_px: int
    total_image_px: int
    coverage_ratio: float   # water / total
    instance_count: int
    mean_confidence: float


def compute_coverage(seg: SegmentationResult) -> CoverageMetrics:
    """Derive coverage metrics from a single-image segmentation result."""
    h, w = seg.image_hw
    total = h * w if h > 0 and w > 0 else 1
    water_px = seg.total_area_px

    confs = [inst.confidence for inst in seg.instances]
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    return CoverageMetrics(
        water_area_px=water_px,
        total_image_px=total,
        coverage_ratio=round(water_px / total, 6),
        instance_count=seg.count,
        mean_confidence=round(mean_conf, 4),
    )


# ── before/after comparison metrics ───────────────────────


@dataclass
class FloodChangeMetrics:
    """Metrics for a before->after flood comparison."""

    before_water_px: int
    after_water_px: int
    permanent_px: int
    flooded_new_px: int
    receded_px: int
    total_image_px: int

    before_coverage: float
    after_coverage: float
    flood_expansion_ratio: float   # new flooding / total
    recession_ratio: float         # receded / total
    net_change_ratio: float        # (after - before) / total  (+ve = expansion)
    iou: float                     # intersection-over-union of water masks

    num_new_regions: int
    num_receded_regions: int


def compute_flood_change(
    before_seg: SegmentationResult,
    after_seg: SegmentationResult,
    comparison: ComparisonResult,
) -> FloodChangeMetrics:
    """Derive change metrics from segmentation results and comparison."""
    before_mask = before_seg.combined_mask
    after_mask = after_seg.combined_mask

    b = (before_mask > 127).astype(np.uint8)
    a = (after_mask > 127).astype(np.uint8)

    b_area = int(b.sum())
    a_area = int(a.sum())
    inter = int((b & a).sum())
    union = int((b | a).sum())
    total = int(b.shape[0] * b.shape[1])

    new_px = int((comparison.flooded_new_mask > 127).sum())
    rec_px = int((comparison.receded_mask > 127).sum())
    perm_px = int((comparison.permanent_mask > 127).sum())

    iou = inter / union if union > 0 else 1.0
    net = (a_area - b_area) / total if total > 0 else 0.0

    n_new = sum(1 for r in comparison.regions if r.label == "flooded_new")
    n_rec = sum(1 for r in comparison.regions if r.label == "receded")

    return FloodChangeMetrics(
        before_water_px=b_area,
        after_water_px=a_area,
        permanent_px=perm_px,
        flooded_new_px=new_px,
        receded_px=rec_px,
        total_image_px=total,
        before_coverage=round(b_area / total, 6) if total else 0.0,
        after_coverage=round(a_area / total, 6) if total else 0.0,
        flood_expansion_ratio=round(new_px / total, 6) if total else 0.0,
        recession_ratio=round(rec_px / total, 6) if total else 0.0,
        net_change_ratio=round(net, 6),
        iou=round(iou, 4),
        num_new_regions=n_new,
        num_receded_regions=n_rec,
    )
