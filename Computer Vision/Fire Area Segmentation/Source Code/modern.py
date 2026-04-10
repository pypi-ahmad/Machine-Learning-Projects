"""Fire Area Segmentation — CVProject registry entry."""

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


@register("fire_area_segmentation")
class FireAreaModern(CVProject):
    """Segment fire and smoke regions with per-frame metrics and trends."""

    project_type = "segmentation"
    description = "Segment fire/smoke regions, estimate area, track trends"
    legacy_tech = "Manual fire monitoring / threshold-based detection"
    modern_tech = "YOLO26m-seg instance segmentation"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import FireController

        self._ctrl = FireController()
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
            "fire_count": m.fire_count,
            "smoke_count": m.smoke_count,
            "fire_area_px": m.fire_area_px,
            "smoke_area_px": m.smoke_area_px,
            "fire_coverage": m.fire_coverage,
            "smoke_coverage": m.smoke_coverage,
            "alert_level": result.alert.level.value,
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

        from visualize import draw_fire_overlay

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        return draw_fire_overlay(
            frame, result.segmentation, result.metrics, self._ctrl.cfg,
        )

    def setup(self, **kwargs) -> None:
        from config import FireConfig
        from controller import FireController

        cfg = FireConfig.from_dict(kwargs) if kwargs else FireConfig()
        self._ctrl = FireController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
