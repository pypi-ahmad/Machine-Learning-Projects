"""Product Counterfeit Visual Checker — CVProject registry entry."""

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


@register("product_counterfeit_visual_checker")
class CounterfeitCheckerModern(CVProject):
    """Screen product images for visual mismatch risk against references."""

    project_type = "screening"
    description = (
        "Compare suspect product images against approved references using "
        "embedding similarity, region-aware checks, and colour histograms "
        "to flag visual mismatch risk (screening only, not proof)"
    )
    legacy_tech = "Manual visual inspection"
    modern_tech = (
        "EfficientNet-B0 embeddings + region patches + HSV histograms"
    )

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import CounterfeitController

        self._ctrl = CounterfeitController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()

        if isinstance(input_data, str):
            result = self._ctrl.screen(input_data)
        else:
            import cv2
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, input_data)
                result = self._ctrl.screen(f.name)
                Path(f.name).unlink(missing_ok=True)

        return {
            "risk_level": result.risk_level,
            "mismatch_risk_pct": result.mismatch_risk_pct,
            "best_composite": result.best_composite,
            "best_reference": result.best_reference,
            "best_product": result.best_product,
            "comparisons": [
                {
                    "reference_path": d.reference_path,
                    "reference_product": d.reference_product,
                    "global_score": d.global_score,
                    "region_score": d.region_score,
                    "histogram_score": d.histogram_score,
                    "composite_score": d.composite_score,
                }
                for d in result.details
            ],
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2

        from visualize import make_screening_grid

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
            result = self._ctrl.screen(input_data)
        else:
            frame = input_data.copy()
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, frame)
                result = self._ctrl.screen(f.name)
                Path(f.name).unlink(missing_ok=True)

        return make_screening_grid(frame, result)

    def setup(self, **kwargs) -> None:
        from config import CounterfeitConfig
        from controller import CounterfeitController

        cfg = CounterfeitConfig.from_dict(kwargs) if kwargs else CounterfeitConfig()
        self._ctrl = CounterfeitController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from reference_builder import main as build_main

        build_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from evaluate import main as eval_main

        eval_main()
