"""Industrial Scratch / Crack Segmentation -- CVProject registry entry."""

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


@register("industrial_scratch_crack_segmentation")
class IndustrialDefectModern(CVProject):
    """Segment surface defects (scratches/cracks) and estimate severity."""

    project_type = "segmentation"
    description = (
        "Detect and segment surface scratches and cracks with "
        "coverage-based severity estimation"
    )
    legacy_tech = "Manual visual inspection / threshold-based edge detection"
    modern_tech = "YOLO26m-seg instance segmentation + coverage-based severity heuristics"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import DefectController
        self._ctrl = DefectController()
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
            "defect_count": m.defect_count,
            "total_defect_area_px": m.total_defect_area_px,
            "defect_coverage": m.defect_coverage,
            "max_length_px": m.max_length_px,
            "mean_length_px": m.mean_length_px,
            "max_aspect_ratio": m.max_aspect_ratio,
            "mean_confidence": m.mean_confidence,
            "severity": m.severity,
            "instances": [
                {
                    "confidence": inst.confidence,
                    "area_px": inst.area_px,
                    "length_px": inst.length_px,
                    "aspect_ratio": inst.aspect_ratio,
                    "bbox": list(inst.bbox),
                }
                for inst in result.segmentation.instances
            ],
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2
        from visualize import draw_defect_overlay
        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.process(frame)
        return draw_defect_overlay(
            frame, result.segmentation, result.metrics, self._ctrl.cfg,
        )

    def setup(self, **kwargs) -> None:
        from config import DefectConfig
        from controller import DefectController
        cfg = DefectConfig.from_dict(kwargs) if kwargs else DefectConfig()
        self._ctrl = DefectController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys
        _sys.argv = [_sys.argv[0], "--eval"]
        from train import main as eval_main
        eval_main()
