"""Wound Area Measurement — per-image wound metrics."""

from __future__ import annotations

from dataclasses import dataclass

from segmentation import SegmentationResult


@dataclass
class WoundMetrics:
    """Metrics computed from a single image's segmentation result."""

    wound_area_px: int
    total_image_px: int
    wound_coverage: float       # wound / total
    wound_count: int
    mean_confidence: float
    largest_wound_px: int       # area of the largest instance


def compute_wound_metrics(seg: SegmentationResult) -> WoundMetrics:
    """Derive wound metrics from a segmentation result."""
    h, w = seg.image_hw
    total = h * w if h > 0 and w > 0 else 1

    confs = [inst.confidence for inst in seg.instances]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    largest = max((inst.area_px for inst in seg.instances), default=0)

    return WoundMetrics(
        wound_area_px=seg.total_area_px,
        total_image_px=total,
        wound_coverage=round(seg.total_area_px / total, 6),
        wound_count=seg.count,
        mean_confidence=round(mean_conf, 4),
        largest_wound_px=largest,
    )
