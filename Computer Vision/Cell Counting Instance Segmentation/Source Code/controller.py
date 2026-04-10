"""Cell Counting Instance Segmentation — main controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CellConfig
from counting import postprocess
from metrics import CellMetrics, compute_cell_metrics
from segmentation import CellSegmenter, SegmentationResult


@dataclass
class FrameResult:
    """Complete result for one processed image."""

    image: np.ndarray
    raw_segmentation: SegmentationResult
    segmentation: SegmentationResult     # after post-processing
    metrics: CellMetrics


class CellController:
    """High-level controller: segmentation → post-processing → metrics."""

    def __init__(self, config: CellConfig | None = None) -> None:
        self.cfg = config or CellConfig()
        self._segmenter = CellSegmenter(
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
        """Run cell segmentation + counting on one image.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        source : str
            Optional source label.
        """
        raw_seg = self._segmenter.segment(frame)

        seg = postprocess(
            raw_seg,
            min_area_px=self.cfg.min_area_px,
            merge_overlap=self.cfg.merge_overlap,
            watershed_split=self.cfg.watershed_split,
        )

        metrics = compute_cell_metrics(seg)

        return FrameResult(
            image=frame,
            raw_segmentation=raw_seg,
            segmentation=seg,
            metrics=metrics,
        )
