"""Wildlife Species Retrieval -- CVProject registry entry."""

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


@register("wildlife_species_retrieval")
class WildlifeSpeciesRetrieval(CVProject):
    """Retrieve visually similar wildlife species images."""

    project_type = "retrieval"
    description = (
        "Embed wildlife images with a pretrained CNN and retrieve "
        "top-k visually similar species from a cosine-similarity index, "
        "with optional classifier reranking"
    )
    legacy_tech = "Manual field guide comparison"
    modern_tech = "EfficientNet-B0 embeddings + cosine index + optional reranking"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = None

    def load(self) -> None:
        from controller import WildlifeController

        self._ctrl = WildlifeController()
        self._ctrl.load()

    def predict(self, input_data):
        if self._ctrl is None:
            self.load()

        if isinstance(input_data, str):
            result = self._ctrl.query(input_data)
        else:
            import tempfile
            import cv2

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, input_data)
                result = self._ctrl.query(f.name)
                Path(f.name).unlink(missing_ok=True)

        return {
            "top_species": next(iter(result.species_votes), None)
            if result.species_votes else None,
            "top_score": result.top_score,
            "species_votes": result.species_votes,
            "matches": [
                {"path": h.path, "species": h.species, "score": h.score}
                for h in result.hits
            ],
        }
