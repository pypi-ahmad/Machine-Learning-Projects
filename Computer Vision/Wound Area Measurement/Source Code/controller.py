"""Wound Area Measurement — main controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from change_tracker import ChangeEntry, ChangeSummary, ChangeTracker
from config import WoundConfig
from metrics import WoundMetrics, compute_wound_metrics
from segmentation import SegmentationResult, WoundSegmenter


@dataclass
class FrameResult:
    """Complete result for one processed image."""

    image: np.ndarray
    segmentation: SegmentationResult
    metrics: WoundMetrics
    change_entry: ChangeEntry | None


class WoundController:
    """High-level controller: segmentation → metrics → optional change tracking."""

    def __init__(self, config: WoundConfig | None = None) -> None:
        self.cfg = config or WoundConfig()
        self._segmenter = WoundSegmenter(
            model_name=self.cfg.model_name,
            confidence=self.cfg.confidence_threshold,
            iou_threshold=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
        )
        self._tracker = ChangeTracker()

    def load(self) -> None:
        """Eagerly load model weights."""
        self._segmenter.load()

    def close(self) -> None:
        """Release model resources."""
        self._segmenter.close()

    def reset_tracker(self) -> None:
        """Clear the change tracker (e.g. between patients)."""
        self._tracker.reset()

    def process(
        self,
        frame: np.ndarray,
        *,
        source: str = "",
        track: bool = False,
    ) -> FrameResult:
        """Run wound segmentation + metrics on one image.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        source : str
            Optional source label for change tracking.
        track : bool
            If True, record this frame in the change tracker.
        """
        seg = self._segmenter.segment(frame)
        metrics = compute_wound_metrics(seg)

        entry = None
        if track:
            entry = self._tracker.add(metrics, source=source)

        return FrameResult(
            image=frame,
            segmentation=seg,
            metrics=metrics,
            change_entry=entry,
        )

    def change_summary(self) -> ChangeSummary:
        """Return the current change-tracking summary."""
        return self._tracker.summary()
