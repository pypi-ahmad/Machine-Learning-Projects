"""Road Pothole Segmentation — pipeline controller.

Orchestrates: segment → severity → validate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import PotholeConfig
from segmentation import PotholeSegmenter, SegmentationResult
from severity import SeverityReport, assess_severity
from validator import ValidationReport


@dataclass
class PotholeResult:
    """Complete output for a single frame."""

    segmentation: SegmentationResult
    severity: SeverityReport
    report: ValidationReport


class PotholeController:
    """High-level controller for the pothole segmentation pipeline."""

    def __init__(self, cfg: PotholeConfig | None = None) -> None:
        self.cfg = cfg or PotholeConfig()
        self._seg = PotholeSegmenter(
            model_name=self.cfg.model_name,
            confidence=self.cfg.confidence_threshold,
            iou_threshold=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
        )

    def load(self) -> None:
        """Eagerly load the segmentation model."""
        self._seg.load()

    def process(self, frame: np.ndarray) -> PotholeResult:
        """Run the full pipeline on a single frame (BGR, uint8).

        Returns
        -------
        PotholeResult
        """
        report = ValidationReport()

        if frame is None or frame.size == 0:
            report.fail("Empty frame")
            return PotholeResult(
                segmentation=SegmentationResult(),
                severity=SeverityReport(),
                report=report,
            )

        seg = self._seg.segment(frame)

        sev = assess_severity(
            seg,
            thresholds=self.cfg.severity_thresholds,
            pixel_area_per_m2=self.cfg.pixel_area_per_m2,
        )

        if seg.count == 0:
            report.warn("No potholes detected")

        return PotholeResult(
            segmentation=seg,
            severity=sev,
            report=report,
        )

    def close(self) -> None:
        self._seg.close()
