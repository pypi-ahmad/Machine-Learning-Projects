"""Industrial Scratch / Crack Segmentation — per-image defect metrics.

All severity proxies are **heuristic** and transparent:

*  ``defect_coverage`` — fraction of image occupied by defects.
*  ``severity`` — categorical label derived from coverage thresholds.
*  ``max_length_px`` — longest defect in pixels (major axis).
*  ``max_aspect_ratio`` — highest length/width ratio (elongation proxy).
"""

from __future__ import annotations

from dataclasses import dataclass

from segmentation import SegmentationResult


@dataclass
class DefectMetrics:
    """Metrics computed from a single image's segmentation result."""

    defect_count: int
    total_defect_area_px: int
    total_image_px: int
    defect_coverage: float          # total_defect_area / total_image
    max_length_px: float            # longest defect
    mean_length_px: float
    max_aspect_ratio: float         # most elongated defect
    mean_confidence: float
    severity: str                   # "none" | "low" | "medium" | "high"


def compute_defect_metrics(
    seg: SegmentationResult,
    *,
    severity_low: float = 0.005,
    severity_medium: float = 0.02,
    min_area_px: int = 32,
) -> DefectMetrics:
    """Derive defect metrics from a segmentation result.

    Parameters
    ----------
    seg : SegmentationResult
        Raw segmentation output.
    severity_low, severity_medium : float
        Coverage thresholds for the ``severity`` label.
    min_area_px : int
        Instances with area below this value are ignored.
    """
    # Filter small detections
    instances = [d for d in seg.instances if d.area_px >= min_area_px]

    h, w = seg.image_hw
    total = h * w if h > 0 and w > 0 else 1

    if not instances:
        return DefectMetrics(
            defect_count=0,
            total_defect_area_px=0,
            total_image_px=total,
            defect_coverage=0.0,
            max_length_px=0.0,
            mean_length_px=0.0,
            max_aspect_ratio=0.0,
            mean_confidence=0.0,
            severity="none",
        )

    total_area = seg.total_area_px
    coverage = total_area / total

    lengths = [d.length_px for d in instances]
    confs = [d.confidence for d in instances]
    ars = [d.aspect_ratio for d in instances]

    if coverage <= severity_low:
        severity = "low"
    elif coverage <= severity_medium:
        severity = "medium"
    else:
        severity = "high"

    return DefectMetrics(
        defect_count=len(instances),
        total_defect_area_px=total_area,
        total_image_px=total,
        defect_coverage=round(coverage, 6),
        max_length_px=round(max(lengths), 1),
        mean_length_px=round(sum(lengths) / len(lengths), 1),
        max_aspect_ratio=round(max(ars), 2),
        mean_confidence=round(sum(confs) / len(confs), 4),
        severity=severity,
    )
