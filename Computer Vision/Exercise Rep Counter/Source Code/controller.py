"""Exercise Rep Counter -- pipeline orchestrator."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from config import ExerciseConfig
from exercise_rules import ExerciseAnalysis, get_exercise_analyser
from pose_detector import PoseDetector, PoseResult
from rep_counter import RepCounter, RepState
from smoother import AngleSmoother
from validator import FrameValidator, ValidationReport


@dataclasses.dataclass
class ControllerResult:
    """Full result from one frame processed by the pipeline."""

    pose: PoseResult
    analysis: ExerciseAnalysis | None
    rep_state: RepState | None
    report: ValidationReport


class ExerciseController:
    """Orchestrates: detect -> analyse exercise -> smooth -> count reps."""

    def __init__(self, config: ExerciseConfig | None = None) -> None:
        self.cfg = config or ExerciseConfig()
        self.detector = PoseDetector(
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            static_image_mode=self.cfg.static_image_mode,
        )
        self._analyse_fn = get_exercise_analyser(self.cfg.exercise)
        self._smoother = AngleSmoother(alpha=self.cfg.ema_alpha)
        self._rep_counter = RepCounter(stable_frames=self.cfg.stable_frames)
        self._validator = FrameValidator()

    def load(self) -> None:
        self.detector.load()

    @property
    def ready(self) -> bool:
        return self.detector.ready

    def process(self, frame: np.ndarray, side: str = "left") -> ControllerResult:
        """Run the full pipeline on a single BGR frame."""
        if not self.ready:
            self.load()

        pose = self.detector.detect(frame)

        if not pose.detected:
            report = self._validator.validate(pose)
            return ControllerResult(
                pose=pose,
                analysis=None,
                rep_state=None,
                report=report,
            )

        # Exercise-specific analysis
        thresholds = self._get_thresholds()
        analysis = self._analyse_fn(pose, *thresholds, side=side)

        # Smooth the raw angle
        if self.cfg.enable_smoothing:
            smoothed_angle = self._smoother.smooth(analysis.angle)
            # Re-derive stage from smoothed angle
            analysis = dataclasses.replace(analysis, angle=smoothed_angle)
            analysis = self._reclassify_stage(analysis, *thresholds)

        # Rep counting
        rep_state = self._rep_counter.update(analysis)

        report = self._validator.validate(
            pose, landmark_indices=analysis.landmarks_used,
        )

        return ControllerResult(
            pose=pose,
            analysis=analysis,
            rep_state=rep_state,
            report=report,
        )

    def _get_thresholds(self) -> tuple[float, float]:
        """Return (down_angle, up_angle) for the current exercise."""
        ex = self.cfg.exercise
        if ex == "squat":
            return self.cfg.squat_down_angle, self.cfg.squat_up_angle
        elif ex == "pushup":
            return self.cfg.pushup_down_angle, self.cfg.pushup_up_angle
        elif ex == "bicep_curl":
            return self.cfg.curl_down_angle, self.cfg.curl_up_angle
        return 90.0, 160.0

    def _reclassify_stage(
        self,
        analysis: ExerciseAnalysis,
        down_threshold: float,
        up_threshold: float,
    ) -> ExerciseAnalysis:
        """Re-derive stage from a (possibly smoothed) angle."""
        invert = analysis.exercise == "bicep_curl"
        angle = analysis.angle

        if invert:
            if angle <= up_threshold:
                stage = "up"
            elif angle >= down_threshold:
                stage = "down"
            else:
                stage = "unknown"
        else:
            if angle <= down_threshold:
                stage = "down"
            elif angle >= up_threshold:
                stage = "up"
            else:
                stage = "unknown"

        return dataclasses.replace(analysis, stage=stage)

    @property
    def reps(self) -> int:
        return self._rep_counter.reps

    def reset(self) -> None:
        self._rep_counter.reset()
        self._smoother.reset()

    def close(self) -> None:
        self.detector.close()
