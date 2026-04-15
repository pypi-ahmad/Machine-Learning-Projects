"""Fire Area Segmentation -- main controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alert import AlertState, evaluate_alert
from config import FireConfig
from metrics import FrameMetrics, compute_frame_metrics
from segmentation import FireSegmenter, SegmentationResult
from trend import TrendSummary, TrendTracker


@dataclass
class FrameResult:
    """Complete result for one processed frame."""

    image: np.ndarray
    segmentation: SegmentationResult
    metrics: FrameMetrics
    trend: TrendSummary
    alert: AlertState


class FireController:
    """High-level controller: segmentation -> metrics -> trend -> alert."""

    def __init__(self, config: FireConfig | None = None) -> None:
        self.cfg = config or FireConfig()
        self._segmenter = FireSegmenter(
            model_name=self.cfg.model_name,
            confidence=self.cfg.confidence_threshold,
            iou_threshold=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            class_names=self.cfg.class_names,
            enable_smoke=self.cfg.enable_smoke,
        )
        self._trend = TrendTracker(window=self.cfg.trend_window)

    def load(self) -> None:
        """Eagerly load model weights."""
        self._segmenter.load()

    def close(self) -> None:
        """Release model resources."""
        self._segmenter.close()

    def reset_trend(self) -> None:
        """Clear trend history (e.g. between videos)."""
        self._trend.reset()

    def process(self, frame: np.ndarray) -> FrameResult:
        """Run the full pipeline on one frame."""
        seg = self._segmenter.segment(frame)
        metrics = compute_frame_metrics(seg)
        trend = self._trend.update(metrics)
        alert = evaluate_alert(metrics, trend)

        return FrameResult(
            image=frame,
            segmentation=seg,
            metrics=metrics,
            trend=trend,
            alert=alert,
        )
