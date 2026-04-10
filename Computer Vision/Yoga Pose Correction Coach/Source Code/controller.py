"""Yoga Pose Correction Coach — pipeline orchestrator."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from config import YogaConfig
from correction_engine import CorrectionHint, generate_corrections
from pose_classifier import ClassificationResult, classify_pose
from pose_detector import PoseDetector, PoseResult
from smoother import PoseSmoother
from validator import FrameValidator, ValidationReport


@dataclasses.dataclass
class CoachResult:
    """Full result from one frame processed by the pipeline."""

    pose: PoseResult
    classification: ClassificationResult | None
    smoothed_pose: str
    corrections: list[CorrectionHint]
    report: ValidationReport


class YogaCoachController:
    """Orchestrates: detect → classify → smooth → correct → validate."""

    def __init__(self, config: YogaConfig | None = None) -> None:
        self.cfg = config or YogaConfig()
        self.detector = PoseDetector(
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            static_image_mode=self.cfg.static_image_mode,
        )
        self._smoother = PoseSmoother(window=self.cfg.vote_window)
        self._validator = FrameValidator(min_visibility=self.cfg.min_visibility)

    def load(self) -> None:
        self.detector.load()

    @property
    def ready(self) -> bool:
        return self.detector.ready

    def process(self, frame: np.ndarray) -> CoachResult:
        """Run the full pipeline on a single BGR frame."""
        if not self.ready:
            self.load()

        pose = self.detector.detect(frame)

        if not pose.detected:
            report = self._validator.validate(pose)
            return CoachResult(
                pose=pose,
                classification=None,
                smoothed_pose="unknown",
                corrections=[],
                report=report,
            )

        # Classify
        cls = classify_pose(pose, confidence_threshold=self.cfg.confidence_threshold)

        # Smooth
        if self.cfg.enable_smoothing:
            smoothed = self._smoother.update(cls.pose)
        else:
            smoothed = cls.pose

        # Generate corrections for the identified pose
        corrections = []
        if smoothed != "unknown":
            corrections = generate_corrections(
                smoothed, cls.angles,
                tolerance=self.cfg.angle_tolerance,
                max_hints=self.cfg.max_hints,
            )

        report = self._validator.validate(pose)

        return CoachResult(
            pose=pose,
            classification=cls,
            smoothed_pose=smoothed,
            corrections=corrections,
            report=report,
        )

    def reset(self) -> None:
        self._smoother.reset()

    def close(self) -> None:
        self.detector.close()
