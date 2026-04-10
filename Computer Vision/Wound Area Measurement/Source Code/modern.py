"""Wound Area Measurement — CVProject registry entry."""

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


@register("wound_area_measurement")
class WoundAreaModern(CVProject):
    """Segment wound regions and estimate relative wound area."""

    project_type = "segmentation"
    description = "Segment wound regions and estimate relative area from images"
    legacy_tech = "Manual wound tracing / ruler-based measurement"
    modern_tech = "YOLO26m-seg instance segmentation"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import WoundController

        self._ctrl = WoundController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()
        import cv2

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data
        result = self._ctrl.process(frame)
        m = result.metrics
        return {
            "wound_count": m.wound_count,
            "wound_area_px": m.wound_area_px,
            "wound_coverage": m.wound_coverage,
            "mean_confidence": m.mean_confidence,
            "largest_wound_px": m.largest_wound_px,
            "instances": [
                {
                    "confidence": inst.confidence,
                    "area_px": inst.area_px,
                    "bbox": list(inst.bbox),
                }
                for inst in result.segmentation.instances
            ],
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2

        from visualize import draw_wound_overlay

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        return draw_wound_overlay(
            frame, result.segmentation, result.metrics, self._ctrl.cfg,
        )

    def setup(self, **kwargs) -> None:
        from config import WoundConfig
        from controller import WoundController

        cfg = WoundConfig.from_dict(kwargs) if kwargs else WoundConfig()
        self._ctrl = WoundController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
