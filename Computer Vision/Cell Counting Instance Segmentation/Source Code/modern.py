"""Cell Counting Instance Segmentation — CVProject registry entry."""

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


@register("cell_counting_instance_segmentation")
class CellCountingModern(CVProject):
    """Segment cells/nuclei and count instances in microscopy images."""

    project_type = "segmentation"
    description = "Segment and count cells/nuclei with post-processing for touching objects"
    legacy_tech = "Manual cell counting / threshold-based segmentation"
    modern_tech = "YOLO26m-seg instance segmentation + watershed splitting"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import CellController

        self._ctrl = CellController()
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
            "cell_count": m.cell_count,
            "total_cell_area_px": m.total_cell_area_px,
            "cell_coverage": m.cell_coverage,
            "mean_cell_area_px": m.mean_cell_area_px,
            "median_cell_area_px": m.median_cell_area_px,
            "mean_confidence": m.mean_confidence,
            "raw_count": result.raw_segmentation.count,
            "instances": [
                {
                    "confidence": inst.confidence,
                    "area_px": inst.area_px,
                    "centroid": list(inst.centroid),
                    "bbox": list(inst.bbox),
                }
                for inst in result.segmentation.instances
            ],
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2

        from visualize import draw_cell_overlay, draw_count_badge

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        vis = draw_cell_overlay(
            frame, result.segmentation, result.metrics, self._ctrl.cfg,
        )
        return draw_count_badge(vis, result.metrics.cell_count)

    def setup(self, **kwargs) -> None:
        from config import CellConfig
        from controller import CellController

        cfg = CellConfig.from_dict(kwargs) if kwargs else CellConfig()
        self._ctrl = CellController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
