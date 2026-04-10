"""Similar Image Finder — retrieval engine.

Combines the embedder and index to answer visual similarity queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import SimilarityConfig

from embedder import ImageEmbedder
from index import ImageIndex, SearchHit


@dataclass
class RetrievalResult:
    """Full retrieval result for one query."""

    query_path: str | None
    query_embedding: np.ndarray
    hits: list[SearchHit]

    @property
    def top_path(self) -> str | None:
        return self.hits[0].path if self.hits else None

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def category_votes(self) -> dict[str, float]:
        """Aggregate similarity by category."""
        votes: dict[str, float] = {}
        for h in self.hits:
            cat = h.category or "unknown"
            votes[cat] = votes.get(cat, 0.0) + h.score
        return dict(sorted(votes.items(), key=lambda kv: kv[1], reverse=True))


class ImageRetriever:
    """Top-k visual similarity retrieval from an embedding index."""

    def __init__(self, cfg: SimilarityConfig | None = None) -> None:
        if cfg is None:
            from config import SimilarityConfig
            cfg = SimilarityConfig()
        self.cfg = cfg
        self._embedder = ImageEmbedder(cfg)
        self._index: ImageIndex | None = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self, index_path: str | None = None) -> None:
        self._embedder.load()
        idx_path = index_path or self.cfg.index_path
        self._index = ImageIndex.load(idx_path)

    def close(self) -> None:
        self._embedder.close()
        self._index = None

    @property
    def index(self) -> ImageIndex | None:
        return self._index

    # ── query API ─────────────────────────────────────────

    def query_image(
        self,
        image_bgr: np.ndarray,
        top_k: int | None = None,
        source: str | None = None,
        exclude_self: bool = True,
    ) -> RetrievalResult:
        if self._index is None:
            raise RuntimeError("Call load() first.")

        embedding = self._embedder.embed(image_bgr)
        k = top_k or self.cfg.top_k
        hits = self._index.search(
            embedding, top_k=k,
            min_similarity=self.cfg.min_similarity,
            exclude_self=exclude_self,
            self_path=source,
        )

        return RetrievalResult(
            query_path=source,
            query_embedding=embedding,
            hits=hits,
        )
