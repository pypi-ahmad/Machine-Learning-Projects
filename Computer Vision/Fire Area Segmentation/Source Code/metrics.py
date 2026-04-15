"""Fire Area Segmentation -- per-frame region metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segmentation import SegmentationResult


@dataclass
class FrameMetrics:
    """Metrics computed from a single frame's segmentation result."""

    fire_area_px: int
    smoke_area_px: int
    total_image_px: int
    fire_coverage: float        # fire / total
    smoke_coverage: float       # smoke / total
    fire_count: int
    smoke_count: int
    mean_fire_conf: float
    mean_smoke_conf: float


def compute_frame_metrics(seg: SegmentationResult) -> FrameMetrics:
    """Derive per-frame metrics from a segmentation result."""
    h, w = seg.image_hw
    total = h * w if h > 0 and w > 0 else 1

    fire_confs = [i.confidence for i in seg.instances if i.class_name == "fire"]
    smoke_confs = [i.confidence for i in seg.instances if i.class_name == "smoke"]

    return FrameMetrics(
        fire_area_px=seg.fire_area_px,
        smoke_area_px=seg.smoke_area_px,
        total_image_px=total,
        fire_coverage=round(seg.fire_area_px / total, 6),
        smoke_coverage=round(seg.smoke_area_px / total, 6),
        fire_count=seg.fire_count,
        smoke_count=seg.smoke_count,
        mean_fire_conf=round(sum(fire_confs) / len(fire_confs), 4) if fire_confs else 0.0,
        mean_smoke_conf=round(sum(smoke_confs) / len(smoke_confs), 4) if smoke_confs else 0.0,
    )
