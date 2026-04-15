"""Finger Counter Pro -- pipeline orchestrator."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from config import FingerCounterConfig
from finger_counter import FingerCounter, FrameCount
from hand_detector import HandDetector, MultiHandResult
from smoother import CountSmoother
from validator import FrameValidator, ValidationReport


@dataclasses.dataclass
class ControllerResult:
    """Full result from one frame processed by the pipeline."""

    multi: MultiHandResult
    frame_count: FrameCount
    smoothed_per_hand: dict[str, int]
    smoothed_total: int
    report: ValidationReport


class CountingController:
    """Orchestrates: detect -> count -> smooth -> validate."""

    def __init__(self, config: FingerCounterConfig | None = None) -> None:
        self.cfg = config or FingerCounterConfig()
        self.detector = HandDetector(
            max_num_hands=self.cfg.max_num_hands,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self.counter = FingerCounter(self.cfg.finger_up_margin)
        self.smoother = CountSmoother(
            alpha=self.cfg.ema_alpha,
            window=self.cfg.vote_window,
        )
        self.validator = FrameValidator()

    def load(self) -> None:
        self.detector.load()

    @property
    def ready(self) -> bool:
        return self.detector.ready

    def process(self, frame: np.ndarray) -> ControllerResult:
        """Run the full pipeline on a single BGR frame."""
        if not self.ready:
            self.load()

        multi = self.detector.detect(frame)
        frame_count = self.counter.analyse_frame(multi.hands)

        if self.cfg.enable_smoothing:
            sm_per, sm_total = self.smoother.update(frame_count)
        else:
            sm_per = {s.handedness: s.finger_count for s in frame_count.per_hand}
            sm_total = frame_count.total

        report = self.validator.validate(multi)

        return ControllerResult(
            multi=multi,
            frame_count=frame_count,
            smoothed_per_hand=sm_per,
            smoothed_total=sm_total,
            report=report,
        )

    def reset(self) -> None:
        self.smoother.reset()

    def close(self) -> None:
        self.detector.close()
