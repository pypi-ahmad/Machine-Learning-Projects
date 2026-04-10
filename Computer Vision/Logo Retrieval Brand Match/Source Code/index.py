"""Logo Retrieval Brand Match — embedding index.

Stores and queries a numpy-based embedding index with brand labels
and file paths.  Supports build, save, load, update, and nearest-
neighbour search via cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class IndexEntry:
    """Metadata for one indexed logo."""

    path: str
    brand: str
    embedding: np.ndarray  # (D,) float32, L2-normalised


@dataclass
class SearchHit:
    """One retrieval result."""

    path: str
    brand: str
    score: float     # cosine similarity
    rank: int


class LogoIndex:
    """Numpy-based embedding index with cosine similarity search."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None   # (N, D)
        self._paths: list[str] = []
        self._brands: list[str] = []

    # ── properties ────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._paths)

    @property
    def embedding_dim(self) -> int:
        if self._embeddings is None:
            return 0
        return self._embeddings.shape[1]

    @property
    def brands(self) -> list[str]:
        return sorted(set(self._brands))

    # ── build / update ────────────────────────────────────

    def add(self, entry: IndexEntry) -> None:
        """Add a single entry to the index."""
        vec = entry.embedding.reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = vec
        else:
            self._embeddings = np.vstack([self._embeddings, vec])
        self._paths.append(entry.path)
        self._brands.append(entry.brand)

    def add_batch(
        self,
        embeddings: np.ndarray,
        paths: list[str],
        brands: list[str],
    ) -> None:
        """Add a batch of entries."""
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        self._paths.extend(paths)
        self._brands.extend(brands)

    # ── search ────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[SearchHit]:
        """Return top-k most similar entries by cosine similarity."""
        if self._embeddings is None or self.size == 0:
            return []

        query = query.reshape(1, -1)
        # cosine similarity (vectors are already L2-normalised)
        sims = (self._embeddings @ query.T).squeeze()

        k = min(top_k, self.size)
        top_indices = np.argsort(sims)[::-1][:k]

        hits: list[SearchHit] = []
        for rank, idx in enumerate(top_indices):
            score = float(sims[idx])
            if score < min_similarity:
                continue
            hits.append(SearchHit(
                path=self._paths[idx],
                brand=self._brands[idx],
                score=score,
                rank=rank + 1,
            ))
        return hits

    # ── persistence ───────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save index to a .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(p),
            embeddings=self._embeddings if self._embeddings is not None else np.empty((0, 0)),
            paths=np.array(self._paths, dtype=object),
            brands=np.array(self._brands, dtype=object),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> LogoIndex:
        """Load index from a .npz file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index not found: {p}")
        data = np.load(str(p), allow_pickle=True)

        idx = cls()
        emb = data["embeddings"]
        if emb.size > 0:
            idx._embeddings = emb.astype(np.float32)
        idx._paths = list(data["paths"])
        idx._brands = list(data["brands"])
        return idx

    # ── stats ─────────────────────────────────────────────

    def summary(self) -> dict:
        brand_counts: dict[str, int] = {}
        for b in self._brands:
            brand_counts[b] = brand_counts.get(b, 0) + 1
        return {
            "total_entries": self.size,
            "embedding_dim": self.embedding_dim,
            "num_brands": len(brand_counts),
            "brand_counts": brand_counts,
        }
