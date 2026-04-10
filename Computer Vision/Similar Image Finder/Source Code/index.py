"""Similar Image Finder — embedding index.

Stores embeddings with file paths and optional category labels.
Supports build, save, load, update, and nearest-neighbour search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    """One search result."""

    path: str
    category: str
    score: float       # cosine similarity (higher = more similar)
    rank: int


class ImageIndex:
    """Numpy-based embedding index with cosine similarity search."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None   # (N, D)
        self._paths: list[str] = []
        self._categories: list[str] = []

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
    def categories(self) -> list[str]:
        return sorted(set(self._categories))

    # ── build / update ────────────────────────────────────

    def add(
        self,
        embedding: np.ndarray,
        path: str,
        category: str = "",
    ) -> None:
        vec = embedding.reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = vec
        else:
            self._embeddings = np.vstack([self._embeddings, vec])
        self._paths.append(path)
        self._categories.append(category)

    def add_batch(
        self,
        embeddings: np.ndarray,
        paths: list[str],
        categories: list[str] | None = None,
    ) -> None:
        if categories is None:
            categories = [""] * len(paths)
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        self._paths.extend(paths)
        self._categories.extend(categories)

    # ── search ────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        top_k: int = 8,
        min_similarity: float = 0.0,
        exclude_self: bool = False,
        self_path: str | None = None,
    ) -> list[SearchHit]:
        """Return top-k most similar entries by cosine similarity."""
        if self._embeddings is None or self.size == 0:
            return []

        q = query.reshape(1, -1)
        sims = (self._embeddings @ q.T).squeeze()

        if exclude_self and self_path:
            for i, p in enumerate(self._paths):
                if p == self_path:
                    sims[i] = -1.0

        k = min(top_k, self.size)
        top_idx = np.argsort(sims)[::-1][:k]

        hits = []
        for rank, idx in enumerate(top_idx):
            score = float(sims[idx])
            if score < min_similarity:
                continue
            hits.append(SearchHit(
                path=self._paths[idx],
                category=self._categories[idx],
                score=score,
                rank=rank + 1,
            ))
        return hits

    # ── persistence ───────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(p),
            embeddings=self._embeddings if self._embeddings is not None else np.empty((0, 0)),
            paths=np.array(self._paths, dtype=object),
            categories=np.array(self._categories, dtype=object),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> ImageIndex:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index not found: {p}")
        data = np.load(str(p), allow_pickle=True)
        idx = cls()
        emb = data["embeddings"]
        if emb.size > 0:
            idx._embeddings = emb.astype(np.float32)
        idx._paths = list(data["paths"])
        idx._categories = list(data["categories"])
        return idx

    # ── stats ─────────────────────────────────────────────

    def summary(self) -> dict:
        cat_counts: dict[str, int] = {}
        for c in self._categories:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        return {
            "total_entries": self.size,
            "embedding_dim": self.embedding_dim,
            "num_categories": len(cat_counts),
            "category_counts": cat_counts,
        }
