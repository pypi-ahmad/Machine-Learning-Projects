"""Finger Counter Pro -- CVProject registry entry."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure shared core is importable
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from core.base import CVProject  # noqa: E402
from core.registry import register  # noqa: E402

# Ensure local package is importable
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@register("finger_counter_pro")
class FingerCounterModern(CVProject):
    """Robust multi-hand finger counting with MediaPipe Hand Landmarker."""

    project_type = "detection"

    def __init__(self) -> None:
        self._ctrl = None
        self._validator = None

    # -- CVProject API ---------------------------------------------------

    def load(self) -> None:
        from controller import CountingController
        from validator import FrameValidator

        self._ctrl = CountingController()
        self._ctrl.load()
        self._validator = FrameValidator()

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
            "per_hand": [
                {
                    "handedness": s.handedness,
                    "finger_count": s.finger_count,
                    "fingers_up": s.fingers_up,
                    "names_up": s.names_up,
                    "confidence": s.confidence,
                }
                for s in r.frame_count.per_hand
            ],
            "total_raw": r.frame_count.total,
            "total_smoothed": r.smoothed_total,
            "hand_count": r.multi.count,
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
        return draw_overlay(
            frame,
            r.multi,
            r.frame_count.per_hand,
            r.smoothed_per_hand,
            r.smoothed_total,
            self._ctrl.detector,
        )

    def setup(self, **kwargs) -> None:
        from config import FingerCounterConfig
        from controller import CountingController

        cfg = FingerCounterConfig.from_dict(kwargs) if kwargs else FingerCounterConfig()
        self._ctrl = CountingController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
