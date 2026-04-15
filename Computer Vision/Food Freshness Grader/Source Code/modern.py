"""Food Freshness Grader -- CVProject registry entry."""

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


@register("food_freshness_grader")
class FoodFreshnessGraderModern(CVProject):
    """Grade food freshness from images (fresh / stale)."""

    project_type = "classification"
    description = (
        "Classify produce images into 12 freshness classes "
        "(6 produce types × fresh/stale) with confidence scores"
    )
    legacy_tech = "Manual visual inspection"
    modern_tech = "ResNet-18 transfer learning (ImageNet -> freshness)"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import FreshnessController

        self._ctrl = FreshnessController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()

        if isinstance(input_data, str):
            result = self._ctrl.grade(input_data)
        else:
            import cv2
            from grader import FreshnessGrader

            grader = self._ctrl._grader
            result = grader.grade(input_data)

        return {
            "freshness": result.freshness,
            "produce": result.produce,
            "class_name": result.class_name,
            "confidence": result.confidence,
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2

        from visualize import annotate_image

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
            result = self._ctrl.grade(input_data)
        else:
            frame = input_data.copy()
            result = self._ctrl._grader.grade(frame)

        return annotate_image(frame, result)

    def setup(self, **kwargs) -> None:
        from config import FreshnessConfig
        from controller import FreshnessController

        cfg = FreshnessConfig.from_dict(kwargs) if kwargs else FreshnessConfig()
        self._ctrl = FreshnessController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main

        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval", "--data", kwargs.get("data", ".")]
        from evaluate import main as eval_main

        eval_main()
