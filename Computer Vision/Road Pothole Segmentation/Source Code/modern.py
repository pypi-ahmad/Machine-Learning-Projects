"""Road Pothole Segmentation — CVProject registry entry."""

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


@register("road_pothole_segmentation")
class PotholeSegModern(CVProject):
    """Pothole instance segmentation with severity estimation."""

    project_type = "segmentation"
    description = "Segment road potholes and estimate severity"
    legacy_tech = "Manual inspection"
    modern_tech = "YOLO26m-seg + severity heuristics"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import PotholeController

        self._ctrl = PotholeController()
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
        return {
            "total_potholes": result.severity.total_count,
            "minor": result.severity.minor_count,
            "moderate": result.severity.moderate_count,
            "severe": result.severity.severe_count,
            "total_area_px": result.severity.total_area_px,
            "road_condition": result.severity.road_condition,
            "potholes": [
                {
                    "severity": a.severity,
                    "area_px": a.area_px,
                    "confidence": a.confidence,
                    "bbox": list(a.bbox),
                }
                for a in result.severity.assessments
            ],
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2

        from visualize import draw_overlay

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        return draw_overlay(frame, result.segmentation, result.severity)

    def setup(self, **kwargs) -> None:
        from config import PotholeConfig
        from controller import PotholeController

        cfg = PotholeConfig.from_dict(kwargs) if kwargs else PotholeConfig()
        self._ctrl = PotholeController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
