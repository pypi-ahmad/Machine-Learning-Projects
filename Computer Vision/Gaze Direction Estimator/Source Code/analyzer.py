"""High-level pipeline for Gaze Direction Estimator.

Orchestrates: landmark detection → iris location → gaze
classification → smoothing.

Usage::

    from analyzer import GazePipeline

    pipeline = GazePipeline(cfg)
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

from calibrator import CalibrationOffsets, GazeCalibrator
from config import GazeConfig
from gaze_classifier import GazeResult, classify_gaze
from iris_locator import IrisPosition, locate_iris
from landmark_engine import LandmarkEngine, LandmarkResult
from smoother import GazeSmoother, SmoothedState

log = logging.getLogger("gaze.analyzer")


@dataclass
class GazeAnalysisResult:
    """Complete pipeline result for a single frame."""

    landmarks: LandmarkResult = field(default_factory=LandmarkResult)
    iris: IrisPosition = field(default_factory=IrisPosition)
    raw_gaze: GazeResult = field(default_factory=GazeResult)
    smoothed: SmoothedState = field(default_factory=SmoothedState)
    face_detected: bool = False
    direction: str = "CENTER"


class GazePipeline:
    """Gaze direction estimation pipeline.

    landmarks → iris location → classify → smooth
    """

    def __init__(self, cfg: GazeConfig) -> None:
        self.cfg = cfg
        self.detector = LandmarkEngine(cfg)
        self.smoother = GazeSmoother(cfg)
        self.calibrator = GazeCalibrator(cfg)
        self._offsets = CalibrationOffsets()
        self._loaded = False

    def load(self) -> None:
        ok = self.detector.load()
        self._loaded = ok

        # Load calibration if file specified
        if self.cfg.calibration_file:
            self._offsets = GazeCalibrator.load(self.cfg.calibration_file)
            if self._offsets.calibrated:
                log.info("Loaded calibration from %s", self.cfg.calibration_file)

        if ok:
            log.info("Gaze pipeline ready")
        else:
            log.error("Pipeline failed to load -- MediaPipe unavailable")

    def process(self, frame: np.ndarray) -> GazeAnalysisResult:
        """Process a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        GazeAnalysisResult
        """
        if not self._loaded:
            self.load()

        result = GazeAnalysisResult()

        # 1. Detect landmarks
        lm = self.detector.detect(frame)
        result.landmarks = lm
        result.face_detected = lm.detected

        if not lm.detected:
            return result

        # 2. Locate iris
        iris = locate_iris(lm)
        result.iris = iris

        if not iris.detected:
            return result

        # 3. Classify gaze direction
        raw = classify_gaze(
            iris, self.cfg,
            h_offset=self._offsets.h_offset,
            v_offset=self._offsets.v_offset,
        )
        result.raw_gaze = raw

        # 4. Smooth
        smoothed = self.smoother.update(raw.h_ratio, raw.v_ratio, raw.direction)
        result.smoothed = smoothed
        result.direction = smoothed.smoothed_direction

        return result

    def set_offsets(self, offsets: CalibrationOffsets) -> None:
        """Apply calibration offsets."""
        self._offsets = offsets

    def reset(self) -> None:
        """Reset all state."""
        self.smoother.reset()
        self.calibrator.reset()
