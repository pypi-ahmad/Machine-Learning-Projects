"""Polyp Lesion Segmentation — CVProject registry entry."""

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


@register("polyp_lesion_segmentation")
class PolypLesionModern(CVProject):
    """Segment polyp/lesion regions in colonoscopy images."""

    project_type = "segmentation"
    description = "Segment polyp regions with YOLO baseline and optional comparison backends"
    legacy_tech = "Manual polyp annotation / threshold-based detection"
    modern_tech = "YOLO26m-seg instance segmentation + optional MedSAM comparison"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import PolypController

        self._ctrl = PolypController()
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
            "polyp_count": m.polyp_count,
            "polyp_area_px": m.polyp_area_px,
            "polyp_coverage": m.polyp_coverage,
            "mean_confidence": m.mean_confidence,
            "largest_polyp_px": m.largest_polyp_px,
            "backend": result.backend_name,
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

        from visualize import draw_polyp_overlay

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        return draw_polyp_overlay(
            frame, result.segmentation, result.metrics, self._ctrl.cfg,
        )

    def setup(self, **kwargs) -> None:
        from config import PolypConfig
        from controller import PolypController

        cfg = PolypConfig.from_dict(kwargs) if kwargs else PolypConfig()
        self._ctrl = PolypController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
