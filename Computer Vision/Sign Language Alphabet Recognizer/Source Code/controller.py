"""Sign Language Alphabet Recognizer -- inference pipeline orchestrator."""

from __future__ import annotations

import collections
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from classifier import SignClassifier
from config import SignLangConfig
from feature_extractor import extract_features
from hand_detector import HandDetector, HandResult
from validator import FrameValidator, ValidationReport


@dataclasses.dataclass
class PredictionResult:
    """Full result from one frame processed by the pipeline."""

    hand: HandResult | None
    label: str | None
    confidence: float
    smoothed_label: str | None
    report: ValidationReport


class RecognitionController:
    """Orchestrates: detect -> extract features -> classify -> smooth."""

    def __init__(self, config: SignLangConfig | None = None) -> None:
        self.cfg = config or SignLangConfig()
        self.detector = HandDetector(
            max_num_hands=self.cfg.max_num_hands,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            static_image_mode=self.cfg.static_image_mode,
        )
        self.classifier = SignClassifier()
        self.validator = FrameValidator(min_confidence=self.cfg.min_detection_confidence)
        self._vote_buf: collections.deque = collections.deque(maxlen=self.cfg.vote_window)

    def load(self, model_path: str | Path | None = None) -> None:
        self.detector.load()
        mp = Path(model_path) if model_path else Path(self.cfg.model_path)
        if mp.exists():
            self.classifier.load(mp)
        else:
            print(f"[WARN] Model not found at {mp} -- run trainer.py first")

    @property
    def ready(self) -> bool:
        return self.detector.ready and self.classifier.ready

    def process(self, frame: np.ndarray) -> PredictionResult:
        """Run the full pipeline on a single BGR frame."""
        if not self.detector.ready:
            self.detector.load()

        hand = self.detector.detect(frame)
        report = self.validator.validate(hand)

        if hand is None or not self.classifier.ready:
            return PredictionResult(
                hand=None,
                label=None,
                confidence=0.0,
                smoothed_label=None,
                report=report,
            )

        feat = extract_features(
            hand,
            normalise_to_wrist=self.cfg.normalise_to_wrist,
            scale_invariant=self.cfg.scale_invariant,
        )
        label, conf = self.classifier.predict(feat)

        # Majority-vote smoothing
        if self.cfg.enable_smoothing:
            self._vote_buf.append(label)
            smoothed = _majority(self._vote_buf)
        else:
            smoothed = label

        return PredictionResult(
            hand=hand,
            label=label,
            confidence=conf,
            smoothed_label=smoothed,
            report=report,
        )

    def reset(self) -> None:
        self._vote_buf.clear()

    def close(self) -> None:
        self.detector.close()


def _majority(buf: collections.deque) -> str:
    """Return most common label in the buffer."""
    if not buf:
        return ""
    counter = collections.Counter(buf)
    return counter.most_common(1)[0][0]
