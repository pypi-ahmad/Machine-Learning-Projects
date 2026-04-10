"""Building Footprint Change Detector — quantitative change metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diff_engine import DiffResult


@dataclass
class ChangeMetrics:
    """Scalar metrics summarising the before→after change."""

    before_area_px: int
    after_area_px: int
    new_area_px: int
    demolished_area_px: int
    unchanged_area_px: int
    total_image_px: int

    iou: float              # intersection-over-union of before & after masks
    before_coverage: float   # fraction of image covered by buildings before
    after_coverage: float    # fraction of image covered by buildings after
    change_ratio: float      # (new + demolished) / total image area
    growth_ratio: float      # after_area / before_area  (>1 = net growth)

    num_new_regions: int
    num_demolished_regions: int


def compute_metrics(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    diff: DiffResult,
) -> ChangeMetrics:
    """Derive metrics from masks and diff result.

    Parameters
    ----------
    before_mask, after_mask
        Binary uint8 masks (H, W). 255 = building.
    diff
        Pre-computed :class:`DiffResult`.
    """
    b = (before_mask > 127).astype(np.uint8)
    a = (after_mask > 127).astype(np.uint8)

    b_area = int(b.sum())
    a_area = int(a.sum())
    intersection = int((b & a).sum())
    union = int((b | a).sum())

    total = int(b.shape[0] * b.shape[1])
    new_px = int((diff.new_mask > 127).sum())
    demo_px = int((diff.demolished_mask > 127).sum())
    unchanged_px = int((diff.unchanged_mask > 127).sum())

    iou = intersection / union if union > 0 else 1.0
    change_ratio = (new_px + demo_px) / total if total > 0 else 0.0
    growth = a_area / b_area if b_area > 0 else float("inf") if a_area > 0 else 1.0

    n_new = sum(1 for r in diff.regions if r.label == "new")
    n_demo = sum(1 for r in diff.regions if r.label == "demolished")

    return ChangeMetrics(
        before_area_px=b_area,
        after_area_px=a_area,
        new_area_px=new_px,
        demolished_area_px=demo_px,
        unchanged_area_px=unchanged_px,
        total_image_px=total,
        iou=round(iou, 4),
        before_coverage=round(b_area / total, 4) if total else 0.0,
        after_coverage=round(a_area / total, 4) if total else 0.0,
        change_ratio=round(change_ratio, 4),
        growth_ratio=round(growth, 4),
        num_new_regions=n_new,
        num_demolished_regions=n_demo,
    )
