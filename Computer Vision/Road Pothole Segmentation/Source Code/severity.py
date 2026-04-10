"""Road Pothole Segmentation — severity estimation and metrics.

Classifies each pothole instance into severity buckets based on mask
area (in pixels or estimated real-world area).  Methodology is
deliberately transparent:

    severity = f(area_px, thresholds)

No opaque model — just thresholds that can be tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from segmentation import PotholeInstance, SegmentationResult


# ── severity levels ────────────────────────────────────────
SEVERITY_MINOR = "minor"
SEVERITY_MODERATE = "moderate"
SEVERITY_SEVERE = "severe"


@dataclass
class PotholeSeverity:
    """Severity assessment for one pothole instance."""

    instance_id: int
    area_px: int
    area_m2: float               # 0.0 if not calibrated
    severity: str                # minor | moderate | severe
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class SeverityReport:
    """Aggregated severity report for all detected potholes."""

    assessments: list[PotholeSeverity] = field(default_factory=list)
    total_count: int = 0
    minor_count: int = 0
    moderate_count: int = 0
    severe_count: int = 0
    total_area_px: int = 0
    total_area_m2: float = 0.0
    road_condition: str = "unknown"


def classify_severity(
    area_px: int,
    thresholds: tuple[int, int] = (1500, 6000),
) -> str:
    """Classify pothole severity by pixel area.

    Parameters
    ----------
    area_px
        Mask area in pixels.
    thresholds
        ``(minor_max, moderate_max)`` — below ``[0]`` = minor,
        between ``[0]`` and ``[1]`` = moderate, above ``[1]`` = severe.

    Returns
    -------
    str
        One of ``"minor"``, ``"moderate"``, ``"severe"``.
    """
    if area_px < thresholds[0]:
        return SEVERITY_MINOR
    elif area_px < thresholds[1]:
        return SEVERITY_MODERATE
    return SEVERITY_SEVERE


def estimate_area_m2(area_px: int, pixel_area_per_m2: float) -> float:
    """Convert pixel area to square metres given a calibration factor.

    Parameters
    ----------
    area_px
        Mask area in pixels.
    pixel_area_per_m2
        Pixels per square metre.  0 means uncalibrated.

    Returns
    -------
    float
        Area in m².  0.0 if uncalibrated.
    """
    if pixel_area_per_m2 <= 0:
        return 0.0
    return round(area_px / pixel_area_per_m2, 4)


def assess_severity(
    seg_result: SegmentationResult,
    *,
    thresholds: tuple[int, int] = (1500, 6000),
    pixel_area_per_m2: float = 0.0,
) -> SeverityReport:
    """Produce a severity report for all potholes in a frame.

    Parameters
    ----------
    seg_result
        Output of :meth:`PotholeSegmenter.segment`.
    thresholds
        ``(minor_max, moderate_max)`` pixel-area boundaries.
    pixel_area_per_m2
        Calibration factor.  0 = uncalibrated.

    Returns
    -------
    SeverityReport
    """
    assessments: list[PotholeSeverity] = []
    minor = moderate = severe = 0
    total_area = 0
    total_m2 = 0.0

    for idx, inst in enumerate(seg_result.instances):
        sev = classify_severity(inst.area_px, thresholds)
        area_m2 = estimate_area_m2(inst.area_px, pixel_area_per_m2)

        assessments.append(PotholeSeverity(
            instance_id=idx,
            area_px=inst.area_px,
            area_m2=area_m2,
            severity=sev,
            confidence=inst.confidence,
            bbox=inst.bbox,
        ))

        if sev == SEVERITY_MINOR:
            minor += 1
        elif sev == SEVERITY_MODERATE:
            moderate += 1
        else:
            severe += 1

        total_area += inst.area_px
        total_m2 += area_m2

    # Overall road condition
    if severe > 0:
        condition = "poor"
    elif moderate > 1:
        condition = "fair"
    elif minor + moderate > 0:
        condition = "acceptable"
    else:
        condition = "good"

    return SeverityReport(
        assessments=assessments,
        total_count=len(assessments),
        minor_count=minor,
        moderate_count=moderate,
        severe_count=severe,
        total_area_px=total_area,
        total_area_m2=round(total_m2, 4),
        road_condition=condition,
    )
