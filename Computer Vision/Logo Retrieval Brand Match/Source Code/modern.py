"""Logo Retrieval Brand Match — CVProject registry entry."""

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


@register("logo_retrieval_brand_match")
class LogoRetrievalModern(CVProject):
    """Detect logos and retrieve the most similar brand matches."""

    project_type = "retrieval"
    description = (
        "Embed logo images and retrieve top-k brand matches from an index "
        "using cosine similarity"
    )
    legacy_tech = "Manual brand look-up / template matching"
    modern_tech = "EfficientNet-B0 embeddings + cosine similarity index"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import LogoController
        self._ctrl = LogoController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()
        import cv2
        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data
        result = self._ctrl.query(frame, source=str(input_data) if isinstance(input_data, str) else None)
        r = result.retrieval
        return {
            "top_brand": r.top_brand,
            "top_score": r.top_score,
            "brand_votes": r.brand_votes,
            "detection_used": result.detection_used,
            "matches": [
                {
                    "rank": h.rank,
                    "brand": h.brand,
                    "score": h.score,
                    "path": h.path,
                }
                for h in r.hits
            ],
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2
        from visualize import make_result_grid
        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
        else:
            frame = input_data.copy()
        result = self._ctrl.query(frame)
        return make_result_grid(result.image, result.retrieval.hits)

    def setup(self, **kwargs) -> None:
        from config import LogoConfig
        from controller import LogoController
        cfg = LogoConfig.from_dict(kwargs) if kwargs else LogoConfig()
        self._ctrl = LogoController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from index_builder import main as build_main
        build_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys
        _sys.argv = [_sys.argv[0], "--eval"]
        from evaluate import main as eval_main
        eval_main()
