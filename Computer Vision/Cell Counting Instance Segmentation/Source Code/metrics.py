"""Cell Counting Instance Segmentation — per-image cell metrics."""

from __future__ import annotations

from dataclasses import dataclass

from segmentation import SegmentationResult


@dataclass
class CellMetrics:
    """Metrics computed from a single image's segmentation result."""

    cell_count: int
    total_cell_area_px: int
    total_image_px: int
    cell_coverage: float        # total_cell_area / total_image
    mean_cell_area_px: float
    median_cell_area_px: float
    min_cell_area_px: int
    max_cell_area_px: int
    mean_confidence: float


def compute_cell_metrics(seg: SegmentationResult) -> CellMetrics:
    """Derive cell-level metrics from a (post-processed) segmentation."""
    h, w = seg.image_hw
    total = h * w if h > 0 and w > 0 else 1

    areas = sorted(inst.area_px for inst in seg.instances)
    confs = [inst.confidence for inst in seg.instances]

    n = len(areas)
    if n == 0:
        return CellMetrics(
            cell_count=0,
            total_cell_area_px=0,
            total_image_px=total,
            cell_coverage=0.0,
            mean_cell_area_px=0.0,
            median_cell_area_px=0.0,
            min_cell_area_px=0,
            max_cell_area_px=0,
            mean_confidence=0.0,
        )

    total_area = seg.total_area_px
    mean_area = sum(areas) / n
    median_area = float(areas[n // 2]) if n % 2 else (areas[n // 2 - 1] + areas[n // 2]) / 2.0
    mean_conf = sum(confs) / n

    return CellMetrics(
        cell_count=n,
        total_cell_area_px=total_area,
        total_image_px=total,
        cell_coverage=round(total_area / total, 6),
        mean_cell_area_px=round(mean_area, 1),
        median_cell_area_px=round(median_area, 1),
        min_cell_area_px=areas[0],
        max_cell_area_px=areas[-1],
        mean_confidence=round(mean_conf, 4),
    )
