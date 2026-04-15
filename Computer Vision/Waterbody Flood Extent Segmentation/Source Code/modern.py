"""Waterbody & Flood Extent Segmentation -- CVProject registry entry."""

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


@register("waterbody_flood_extent_segmentation")
class WaterFloodModern(CVProject):
    """Detect water bodies and compare flood extent in satellite imagery."""

    project_type = "segmentation"
    description = "Segment water bodies and compare before/after flood extent"
    legacy_tech = "Manual satellite image interpretation"
    modern_tech = "YOLO26m-seg instance segmentation"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import WaterFloodController

        self._ctrl = WaterFloodController()
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
        cov = result.coverage
        return {
            "water_regions": cov.instance_count,
            "water_area_px": cov.water_area_px,
            "coverage_ratio": cov.coverage_ratio,
            "mean_confidence": cov.mean_confidence,
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

        from visualize import draw_water_overlay

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        return draw_water_overlay(
            frame, result.segmentation, result.coverage, self._ctrl.cfg,
        )

    def setup(self, **kwargs) -> None:
        from config import FloodConfig
        from controller import WaterFloodController

        cfg = FloodConfig.from_dict(kwargs) if kwargs else FloodConfig()
        self._ctrl = WaterFloodController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
