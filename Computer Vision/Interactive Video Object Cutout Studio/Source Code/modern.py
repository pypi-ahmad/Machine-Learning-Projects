"""Interactive Video Object Cutout Studio — CVProject registry entry."""

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


@register("interactive_video_object_cutout_studio")
class VideoObjectCutoutModern(CVProject):
    """Promptable object segmentation & video mask propagation with SAM 2."""

    project_type = "segmentation"
    description = (
        "Click or box-prompt an object in an image or video, then "
        "propagate the mask across frames with SAM 2"
    )
    legacy_tech = "Manual rotoscoping / chroma-keying"
    modern_tech = "SAM 2.1 promptable segmentation + video mask propagation"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import CutoutController
        self._ctrl = CutoutController()
        self._ctrl.load_image_engine()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()
        import cv2
        import numpy as np
        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data

        # Auto-segment with a centre-point prompt
        h, w = frame.shape[:2]
        pts = np.array([[w // 2, h // 2]], dtype=np.float32)
        lbls = np.array([1], dtype=np.int32)

        result = self._ctrl.segment_image(frame, points=pts, labels=lbls)
        mr = result.mask_result
        return {
            "best_score": result.score,
            "mask_area_px": int(result.best_mask.sum()),
            "num_candidates": int(mr.masks.shape[0]),
            "scores": mr.scores.tolist(),
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2
        import numpy as np
        from export import draw_overlay
        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()

        h, w = frame.shape[:2]
        pts = np.array([[w // 2, h // 2]], dtype=np.float32)
        lbls = np.array([1], dtype=np.int32)
        result = self._ctrl.segment_image(frame, points=pts, labels=lbls)
        return draw_overlay(frame, result.best_mask)

    def setup(self, **kwargs) -> None:
        from config import CutoutConfig
        from controller import CutoutController
        cfg = CutoutConfig.from_dict(kwargs) if kwargs else CutoutConfig()
        self._ctrl = CutoutController(cfg)
        self._ctrl.load_image_engine()

    def train(self, **kwargs) -> None:
        print("SAM 2 is a foundation model — no project-specific training needed.")
        print("Use fine-tuning scripts from facebookresearch/sam2 if desired.")

    def evaluate(self, **kwargs) -> None:
        import sys as _sys
        _sys.argv = [_sys.argv[0], "--eval"]
        from benchmark import main as bench_main
        bench_main()
