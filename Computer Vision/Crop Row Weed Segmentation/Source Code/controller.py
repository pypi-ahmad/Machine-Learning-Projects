"""Crop Row & Weed Segmentation — pipeline controller.

Orchestrates: segment → class stats → validate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from class_stats import AreaReport, compute_area_stats
from config import CropWeedConfig
from segmentation import CropWeedSegmenter, SegmentationResult
from validator import ValidationReport


@dataclass
class CropWeedResult:
    """Complete output for a single frame."""

    segmentation: SegmentationResult
    area_report: AreaReport
    report: ValidationReport


class CropWeedController:
    """High-level controller for the crop/weed segmentation pipeline."""

    def __init__(self, cfg: CropWeedConfig | None = None) -> None:
        self.cfg = cfg or CropWeedConfig()
        self._seg = CropWeedSegmenter(
            model_name=self.cfg.model_name,
            confidence=self.cfg.confidence_threshold,
            iou_threshold=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            class_names=self.cfg.class_names,
        )

    def load(self) -> None:
        """Eagerly load the segmentation model."""
        self._seg.load()

    def process(self, frame: np.ndarray) -> CropWeedResult:
        """Run the full pipeline on a single frame (BGR, uint8)."""
        report = ValidationReport()

        if frame is None or frame.size == 0:
            report.fail("Empty frame")
            return CropWeedResult(
                segmentation=SegmentationResult(),
                area_report=AreaReport(),
                report=report,
            )

        seg = self._seg.segment(frame)
        area = compute_area_stats(seg, class_names=self.cfg.class_names)

        if seg.count == 0:
            report.warn("No instances detected")

        return CropWeedResult(
            segmentation=seg,
            area_report=area,
            report=report,
        )

    def close(self) -> None:
        self._seg.close()
