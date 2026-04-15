"""Crop Row & Weed Segmentation -- CVProject registry entry."""

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


@register("crop_row_weed_segmentation")
class CropWeedModern(CVProject):
    """Multi-class agricultural segmentation: crop rows, weeds, soil."""

    project_type = "segmentation"
    description = "Segment crop rows, weeds, and soil from agricultural imagery"
    legacy_tech = "Manual field inspection"
    modern_tech = "YOLO26m-seg multi-class"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import CropWeedController

        self._ctrl = CropWeedController()
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
            "total_instances": result.area_report.total_instances,
            "background_ratio": result.area_report.background_ratio,
            "per_class": {
                name: {
                    "count": s.instance_count,
                    "area_px": s.total_area_px,
                    "coverage": s.coverage_ratio,
                }
                for name, s in result.area_report.per_class.items()
            },
            "instances": [
                {
                    "class": inst.class_name,
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

        from visualize import draw_overlay

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        return draw_overlay(frame, result.segmentation, result.area_report, self._ctrl.cfg)

    def setup(self, **kwargs) -> None:
        from config import CropWeedConfig
        from controller import CropWeedController

        cfg = CropWeedConfig.from_dict(kwargs) if kwargs else CropWeedConfig()
        self._ctrl = CropWeedController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
