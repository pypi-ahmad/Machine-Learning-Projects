"""Yoga Pose Correction Coach -- CVProject registry entry."""

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


@register("yoga_pose_correction_coach")
class YogaCoachModern(CVProject):
    """Yoga pose identification with rule-based correction hints."""

    project_type = "detection"

    def __init__(self) -> None:
        self._ctrl = None

    def load(self) -> None:
        from controller import YogaCoachController

        self._ctrl = YogaCoachController()
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
        out = {
            "pose_detected": r.pose.detected,
            "pose_label": r.smoothed_pose,
            "corrections": [
                {"joint": c.joint, "hint": c.hint, "severity": c.severity}
                for c in r.corrections
            ],
            "report": {"ok": r.report.ok, "warnings": r.report.warnings},
        }
        if r.classification:
            out["confidence"] = r.classification.confidence
            out["scores"] = r.classification.scores
        return out

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
        from config import YogaConfig
        from controller import YogaCoachController

        cfg = YogaConfig.from_dict(kwargs) if kwargs else YogaConfig()
        self._ctrl = YogaCoachController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main

        eval_main()
