"""Similar Image Finder — CVProject registry entry."""

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


@register("similar_image_finder")
class SimilarImageFinderModern(CVProject):
    """Find visually similar images from an indexed corpus."""

    project_type = "retrieval"
    description = (
        "Embed images and retrieve top-k visually similar matches from "
        "a cosine-similarity index"
    )
    legacy_tech = "Manual browsing / pixel-level comparison"
    modern_tech = "EfficientNet-B0 embeddings + cosine similarity index"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import SimilarityController

        self._ctrl = SimilarityController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()
        import cv2

        if isinstance(input_data, str):
            result = self._ctrl.query(input_data)
        else:
            # ndarray — save to temp, query, cleanup
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, input_data)
                result = self._ctrl.query(f.name)
                Path(f.name).unlink(missing_ok=True)

        return {
            "top_category": max(result.category_votes, key=result.category_votes.get)
            if result.category_votes
            else None,
            "top_score": result.top_score,
            "category_votes": result.category_votes,
            "matches": [
                {
                    "rank": h.rank,
                    "category": h.category,
                    "score": h.score,
                    "path": h.path,
                }
                for h in result.hits
            ],
        }

    def visualize(self, input_data, output):
        if self._ctrl is None:
            self.load()
        import cv2

        from visualize import make_result_grid

        if isinstance(input_data, str):
            frame = cv2.imread(input_data)
            result = self._ctrl.query(input_data)
        else:
            frame = input_data.copy()
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, frame)
                result = self._ctrl.query(f.name)
                Path(f.name).unlink(missing_ok=True)

        return make_result_grid(frame, result.hits)

    def setup(self, **kwargs) -> None:
        from config import SimilarityConfig
        from controller import SimilarityController

        cfg = SimilarityConfig.from_dict(kwargs) if kwargs else SimilarityConfig()
        self._ctrl = SimilarityController(cfg)
        self._ctrl.load()

    def train(self, **kwargs) -> None:
        from index_builder import main as build_main

        build_main()

    def evaluate(self, **kwargs) -> None:
        import sys as _sys

        _sys.argv = [_sys.argv[0], "--eval"]
        from evaluate import main as eval_main

        eval_main()
