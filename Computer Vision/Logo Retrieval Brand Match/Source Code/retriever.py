"""Logo Retrieval Brand Match — retrieval engine.

Combines the embedder and the index to answer "which brand is this?"
queries.  Keeps detection and retrieval logic separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import LogoConfig

from embedder import LogoEmbedder
from index import LogoIndex, SearchHit


@dataclass
class RetrievalResult:
    """Full retrieval result for one query."""

    query_path: str | None
    query_embedding: np.ndarray
    hits: list[SearchHit]

    @property
    def top_brand(self) -> str | None:
        return self.hits[0].brand if self.hits else None

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def brand_votes(self) -> dict[str, float]:
        """Aggregate similarity by brand (sum of top-k scores)."""
        votes: dict[str, float] = {}
        for h in self.hits:
            votes[h.brand] = votes.get(h.brand, 0.0) + h.score
        return dict(sorted(votes.items(), key=lambda kv: kv[1], reverse=True))


class LogoRetriever:
    """Top-k brand retrieval from an embedding index."""

    def __init__(self, cfg: LogoConfig | None = None) -> None:
        if cfg is None:
            from config import LogoConfig
            cfg = LogoConfig()
        self.cfg = cfg
        self._embedder = LogoEmbedder(cfg)
        self._index: LogoIndex | None = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self, index_path: str | None = None) -> None:
        """Load embedder weights and the pre-built index."""
        self._embedder.load()
        idx_path = index_path or self.cfg.index_path
        self._index = LogoIndex.load(idx_path)

    def close(self) -> None:
        self._embedder.close()
        self._index = None

    @property
    def index(self) -> LogoIndex | None:
        return self._index

    # ── query API ─────────────────────────────────────────

    def query_image(
        self,
        image_bgr: np.ndarray,
        top_k: int | None = None,
        source: str | None = None,
    ) -> RetrievalResult:
        """Embed *image_bgr* and return the top-k brand matches."""
        if self._index is None:
            raise RuntimeError("Call load() first.")

        embedding = self._embedder.embed(image_bgr)
        k = top_k or self.cfg.top_k
        hits = self._index.search(embedding, top_k=k, min_similarity=self.cfg.min_similarity)

        return RetrievalResult(
            query_path=source,
            query_embedding=embedding,
            hits=hits,
        )

    def query_embedding(
        self,
        embedding: np.ndarray,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Query with a precomputed embedding vector."""
        if self._index is None:
            raise RuntimeError("Call load() first.")

        k = top_k or self.cfg.top_k
        hits = self._index.search(embedding, top_k=k, min_similarity=self.cfg.min_similarity)
        return RetrievalResult(
            query_path=None,
            query_embedding=embedding,
            hits=hits,
        )
