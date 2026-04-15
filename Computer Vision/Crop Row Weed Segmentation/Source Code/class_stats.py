"""Crop Row & Weed Segmentation — per-class area statistics.
"""Crop Row & Weed Segmentation — per-class area statistics.

Computes per-class pixel coverage, instance counts, and overall
class distribution from a :class:`SegmentationResult`.
"""
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from segmentation import SegmentationResult


@dataclass
class ClassStats:
    """Statistics for a single class."""

    class_name: str
    instance_count: int = 0
    total_area_px: int = 0
    mean_area_px: float = 0.0
    coverage_ratio: float = 0.0   # fraction of total image area
    mean_confidence: float = 0.0


@dataclass
class AreaReport:
    """Aggregated per-class area statistics for one frame."""

    per_class: dict[str, ClassStats] = field(default_factory=dict)
    total_instances: int = 0
    total_segmented_px: int = 0
    total_image_px: int = 0
    background_ratio: float = 1.0  # fraction not covered by any class


def compute_area_stats(
    seg: SegmentationResult,
    *,
    class_names: list[str] | None = None,
) -> AreaReport:
    """Compute per-class area statistics from segmentation results.
    """Compute per-class area statistics from segmentation results.

    Parameters
    ----------
    seg
        Output of :meth:`CropWeedSegmenter.segment`.
    class_names
        Expected class names.  Classes with zero detections are
        included in the report with zeroed statistics.

    Returns
    -------
    AreaReport
    """
    """
    h, w = seg.image_hw
    total_px = h * w if h > 0 and w > 0 else 1

    # Accumulate per class
    accum: dict[str, list[tuple[int, float]]] = {}  # name -> [(area, conf), ...]
    for inst in seg.instances:
        accum.setdefault(inst.class_name, []).append(
            (inst.area_px, inst.confidence),
        )

    # Ensure all expected classes are present
    if class_names:
        for cn in class_names:
            accum.setdefault(cn, [])

    per_class: dict[str, ClassStats] = {}
    total_seg = 0

    for name, items in sorted(accum.items()):
        count = len(items)
        areas = [a for a, _ in items]
        confs = [c for _, c in items]
        total_area = sum(areas)
        mean_area = total_area / count if count else 0.0
        mean_conf = sum(confs) / count if count else 0.0

        per_class[name] = ClassStats(
            class_name=name,
            instance_count=count,
            total_area_px=total_area,
            mean_area_px=round(mean_area, 1),
            coverage_ratio=round(total_area / total_px, 6),
            mean_confidence=round(mean_conf, 4),
        )
        total_seg += total_area

    # Background = pixels not covered by any instance mask
    # Use combined mask to avoid double-counting overlaps
    if seg.class_masks:
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in seg.class_masks.values():
            combined = np.bitwise_or(combined, m)
        covered = int((combined > 0).sum())
    else:
        covered = 0

    bg_ratio = 1.0 - (covered / total_px) if total_px > 0 else 1.0

    return AreaReport(
        per_class=per_class,
        total_instances=seg.count,
        total_segmented_px=covered,
        total_image_px=total_px,
        background_ratio=round(max(bg_ratio, 0.0), 6),
    )
