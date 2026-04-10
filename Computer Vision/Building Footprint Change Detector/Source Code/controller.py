"""Building Footprint Change Detector — pipeline controller.

Orchestrates: preprocess → segment → diff → metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import ChangeConfig
from diff_engine import DiffResult, compute_diff
from metrics import ChangeMetrics, compute_metrics
from preprocess import ImagePair, prepare_pair
from segmentation import BuildingSegmenter, SegmentationResult
from validator import ValidationReport, validate_pair


@dataclass
class ChangeResult:
    """Complete output for a single before/after comparison."""

    pair: ImagePair
    before_seg: SegmentationResult
    after_seg: SegmentationResult
    diff: DiffResult
    metrics: ChangeMetrics
    report: ValidationReport


class ChangeDetectorController:
    """High-level controller for the change-detection pipeline."""

    def __init__(self, cfg: ChangeConfig | None = None) -> None:
        self.cfg = cfg or ChangeConfig()
        self._seg = BuildingSegmenter(
            model_name=self.cfg.model_name,
            confidence=self.cfg.confidence_threshold,
            iou_threshold=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            use_all_classes=self.cfg.use_all_classes,
        )

    def load(self) -> None:
        """Eagerly load the segmentation model."""
        self._seg.load()

    def process_pair(
        self,
        before_path: str | Path,
        after_path: str | Path,
        *,
        match_histograms: bool = False,
    ) -> ChangeResult:
        """Run the full pipeline on one image pair.

        Parameters
        ----------
        before_path, after_path
            Paths to the before and after images.
        match_histograms
            Apply histogram matching to reduce illumination differences.

        Returns
        -------
        ChangeResult
        """
        report = validate_pair(before_path, after_path)

        pair = prepare_pair(
            before_path, after_path,
            target_size=self.cfg.imgsz,
            match_histograms=match_histograms,
        )

        before_seg = self._seg.segment(pair.before)
        after_seg = self._seg.segment(pair.after)

        diff = compute_diff(
            before_seg.mask,
            after_seg.mask,
            morph_kernel_size=self.cfg.morph_kernel_size,
            min_change_area=self.cfg.min_change_area,
        )

        m = compute_metrics(before_seg.mask, after_seg.mask, diff)

        return ChangeResult(
            pair=pair,
            before_seg=before_seg,
            after_seg=after_seg,
            diff=diff,
            metrics=m,
            report=report,
        )

    def process_images(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> ChangeResult:
        """Run the pipeline on pre-loaded numpy arrays (BGR, uint8)."""
        import cv2
        from preprocess import ImagePair, resize_to_common

        b, a, size = resize_to_common(before, after, self.cfg.imgsz)
        pair = ImagePair(
            before=b, after=a,
            original_before=before, original_after=after,
            target_size=size,
        )

        before_seg = self._seg.segment(pair.before)
        after_seg = self._seg.segment(pair.after)

        diff = compute_diff(
            before_seg.mask, after_seg.mask,
            morph_kernel_size=self.cfg.morph_kernel_size,
            min_change_area=self.cfg.min_change_area,
        )
        m = compute_metrics(before_seg.mask, after_seg.mask, diff)

        return ChangeResult(
            pair=pair,
            before_seg=before_seg,
            after_seg=after_seg,
            diff=diff,
            metrics=m,
            report=ValidationReport(),
        )

    def close(self) -> None:
        self._seg.close()
