"""Waterbody & Flood Extent Segmentation — main controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import FloodConfig
from .coverage import (
    CoverageMetrics,
    FloodChangeMetrics,
    compute_coverage,
    compute_flood_change,
)
from .flood_compare import ComparisonResult, compare_flood_extent
from .segmentation import SegmentationResult, WaterSegmenter

logger = logging.getLogger(__name__)


@dataclass
class SingleResult:
    """Result for a single-image water segmentation."""

    image: np.ndarray
    segmentation: SegmentationResult
    coverage: CoverageMetrics


@dataclass
class PairResult:
    """Result for a before/after flood comparison."""

    before_image: np.ndarray
    after_image: np.ndarray
    before_seg: SegmentationResult
    after_seg: SegmentationResult
    comparison: ComparisonResult
    flood_metrics: FloodChangeMetrics


class WaterFloodController:
    """High-level controller for water/flood segmentation."""

    def __init__(self, config: FloodConfig | None = None) -> None:
        self.cfg = config or FloodConfig()
        self._segmenter = WaterSegmenter(
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

    # ------------------------------------------------------------------
    # Single-image mode
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> SingleResult:
        """Run water segmentation + coverage metrics on one image."""
        seg = self._segmenter.segment(frame)
        cov = compute_coverage(seg)
        return SingleResult(image=frame, segmentation=seg, coverage=cov)

    def process_path(self, path: str | Path) -> SingleResult:
        """Load an image from *path* and process it."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.process(img)

    # ------------------------------------------------------------------
    # Before / after comparison mode
    # ------------------------------------------------------------------

    def process_pair(
        self,
        before_path: str | Path,
        after_path: str | Path,
    ) -> PairResult:
        """Run water segmentation + flood comparison on a pair."""
        before_img = cv2.imread(str(before_path))
        after_img = cv2.imread(str(after_path))
        if before_img is None:
            raise FileNotFoundError(f"Cannot read before image: {before_path}")
        if after_img is None:
            raise FileNotFoundError(f"Cannot read after image: {after_path}")

        before_seg = self._segmenter.segment(before_img)
        after_seg = self._segmenter.segment(after_img)

        comparison = compare_flood_extent(
            before_mask=before_seg.combined_mask,
            after_mask=after_seg.combined_mask,
            morph_kernel_size=self.cfg.morph_kernel_size,
            min_change_area=self.cfg.min_change_area,
        )

        flood_metrics = compute_flood_change(
            before_seg,
            after_seg,
            comparison,
        )

        return PairResult(
            before_image=before_img,
            after_image=after_img,
            before_seg=before_seg,
            after_seg=after_seg,
            comparison=comparison,
            flood_metrics=flood_metrics,
        )
