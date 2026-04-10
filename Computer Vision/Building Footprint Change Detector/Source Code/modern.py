"""Building Footprint Change Detector — CVProject registry entry."""

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


@register("building_footprint_change_detector")
class BuildingChangeModern(CVProject):
    """Before/after aerial analysis for building footprint changes."""

    project_type = "segmentation"
    description = "Detect building footprint changes between before/after aerial images"
    legacy_tech = "Manual comparison"
    modern_tech = "YOLO26m-seg + mask diff"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import ChangeDetectorController

        self._ctrl = ChangeDetectorController()
        self._ctrl.load()

    def predict(self, input_data):
        """Accept a dict ``{"before": path, "after": path}`` or two-element list."""
        if self._ctrl is None:
            self.load()

        if isinstance(input_data, dict):
            before = input_data["before"]
            after = input_data["after"]
        elif isinstance(input_data, (list, tuple)) and len(input_data) == 2:
            before, after = input_data
        else:
            raise ValueError(
                "input_data must be {'before': path, 'after': path} "
                "or [before_path, after_path]"
            )

        import cv2

        if isinstance(before, str):
            before_img = cv2.imread(before)
        else:
            before_img = before
        if isinstance(after, str):
            after_img = cv2.imread(after)
        else:
            after_img = after

        result = self._ctrl.process_images(before_img, after_img)
        return {
            "iou": result.metrics.iou,
            "change_ratio": result.metrics.change_ratio,
            "growth_ratio": result.metrics.growth_ratio,
            "new_regions": result.metrics.num_new_regions,
            "demolished_regions": result.metrics.num_demolished_regions,
            "before_area_px": result.metrics.before_area_px,
            "after_area_px": result.metrics.after_area_px,
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2

        from visualize import compose_report

        if isinstance(input_data, dict):
            before = input_data["before"]
            after = input_data["after"]
        else:
            before, after = input_data

        if isinstance(before, str):
            before = cv2.imread(before)
        if isinstance(after, str):
            after = cv2.imread(after)

        result = self._ctrl.process_images(before, after)
        return compose_report(
            result.pair.before, result.pair.after,
            result.before_seg.mask, result.after_seg.mask,
            result.diff, result.metrics,
        )

    def setup(self, **kwargs) -> None:
        from config import ChangeConfig
        from controller import ChangeDetectorController

        cfg = ChangeConfig.from_dict(kwargs) if kwargs else ChangeConfig()
        self._ctrl = ChangeDetectorController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main

        eval_main()
