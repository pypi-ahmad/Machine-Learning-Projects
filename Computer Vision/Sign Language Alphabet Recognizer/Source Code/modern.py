"""Sign Language Alphabet Recognizer -- CVProject registry entry."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from core.base import CVProject  # noqa: E402
from core.registry import register  # noqa: E402

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@register("sign_language_alphabet_recognizer")
class SignLangModern(CVProject):
    """Static ASL alphabet recognition from hand landmarks."""

    project_type = "classification"

    def __init__(self) -> None:
        self._ctrl = None

    def load(self) -> None:
        from controller import RecognitionController

        self._ctrl = RecognitionController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()
        import cv2

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data
        r = self._ctrl.process(frame)
        return {
            "label": r.label,
            "confidence": r.confidence,
            "smoothed_label": r.smoothed_label,
            "hand_detected": r.hand is not None,
            "report": {"ok": r.report.ok, "warnings": r.report.warnings},
        }

    def visualize(self, input_data, output):
        import cv2
        from visualize import draw_overlay

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        r = self._ctrl.process(frame)
        return draw_overlay(frame, r, self._ctrl.detector)

    def setup(self, **kwargs) -> None:
        from config import SignLangConfig
        from controller import RecognitionController

        cfg = SignLangConfig.from_dict(kwargs) if kwargs else SignLangConfig()
        self._ctrl = RecognitionController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from trainer import main as train_main

        train_main(**kwargs)

    def evaluate(self, **kwargs) -> None:
        from trainer import main as eval_main

        eval_main(**kwargs)
