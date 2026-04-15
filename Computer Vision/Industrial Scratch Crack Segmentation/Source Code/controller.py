"""Industrial Scratch / Crack Segmentation -- main controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DefectConfig
from metrics import DefectMetrics, compute_defect_metrics
from segmentation import DefectSegmenter, SegmentationResult


@dataclass
class FrameResult:
    """Complete result for one processed image."""

    image: np.ndarray
    segmentation: SegmentationResult
    metrics: DefectMetrics


class DefectController:
    """High-level controller: segmentation -> metrics."""

    def __init__(self, config: DefectConfig | None = None) -> None:
        self.cfg = config or DefectConfig()
        self._segmenter = DefectSegmenter(
            model_name=self.cfg.model_name,
            confidence=self.cfg.confidence_threshold,
            iou_threshold=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
        )

    def load(self) -> None:
        """Eagerly load model weights."""
        self._segmenter.load()

    def close(self) -> None:
        """Release model resources."""
        self._segmenter.close()

    def process(
        self,
        frame: np.ndarray,
        *,
        source: str = "",
    ) -> FrameResult:
        """Run defect segmentation + metrics on one image.
        """Run defect segmentation + metrics on one image.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        source : str
            Optional source label.
        """
        """
        seg = self._segmenter.segment(frame)
        metrics = compute_defect_metrics(
            seg,
            severity_low=self.cfg.severity_low,
            severity_medium=self.cfg.severity_medium,
            min_area_px=self.cfg.min_area_px,
        )

        return FrameResult(
            image=frame,
            segmentation=seg,
            metrics=metrics,
        )
