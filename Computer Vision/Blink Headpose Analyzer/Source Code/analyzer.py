"""High-level pipeline for Blink Headpose Analyzer.

Orchestrates: landmark detection → blink counting → head pose.

Usage::

    from analyzer import AnalyzerPipeline

    pipeline = AnalyzerPipeline(cfg)
    pipeline.load()
    result = pipeline.process(frame)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blink_counter import BlinkCounter, BlinkState
from config import AnalyzerConfig
from landmark_engine import LandmarkEngine, LandmarkResult
from pose_estimator import PoseEstimator, PoseState

log = logging.getLogger("blink_headpose.analyzer")


@dataclass
class AnalysisResult:
    """Complete pipeline result for a single frame."""

    landmarks: LandmarkResult = field(default_factory=LandmarkResult)
    blink: BlinkState = field(default_factory=BlinkState)
    head_pose: PoseState = field(default_factory=PoseState)
    face_detected: bool = False


class AnalyzerPipeline:
    """Blink counting + head pose estimation pipeline.

    landmarks → blink EAR → head pose
    """

    def __init__(self, cfg: AnalyzerConfig) -> None:
        self.cfg = cfg
        self.detector = LandmarkEngine(cfg)
        self.blink_counter = BlinkCounter(cfg)
        self.pose_estimator = PoseEstimator(cfg)
        self._loaded = False

    def load(self) -> None:
        ok = self.detector.load()
        self._loaded = ok
        if ok:
            log.info("Analyzer pipeline ready")
        else:
            log.error("Pipeline failed to load -- MediaPipe unavailable")

    def process(self, frame: np.ndarray) -> AnalysisResult:
        """Process a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        AnalysisResult
        """
        if not self._loaded:
            self.load()

        result = AnalysisResult()

        # 1. Detect landmarks
        lm = self.detector.detect(frame)
        result.landmarks = lm
        result.face_detected = lm.detected

        if not lm.detected:
            return result

        # 2. Blink counting
        result.blink = self.blink_counter.update(lm)

        # 3. Head pose estimation
        result.head_pose = self.pose_estimator.update(lm)

        return result

    def reset(self) -> None:
        """Reset all tracker state."""
        self.blink_counter.reset()
