"""Product Counterfeit Visual Checker — reference store.

Manages a collection of approved product reference embeddings stored
in a NumPy .npz archive.  Each entry has a path, product label, and
embedding vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReferenceHit:
    """A single reference match result."""

    path: str
    product: str
    score: float
    rank: int


class ReferenceStore:
    """Approved product reference embeddings with nearest-neighbour lookup."""

    def __init__(self, metric: str = "cosine") -> None:
        self._embeddings: np.ndarray | None = None  # (N, D)
        self._paths: list[str] = []
        self._products: list[str] = []
        self._metric = metric

    def __len__(self) -> int:
        return len(self._paths)

    # ── add ────────────────────────────────────────────────

    def add(
        self,
        embedding: np.ndarray,
        path: str,
        product: str = "",
    ) -> None:
        vec = embedding.reshape(1, -1).astype(np.float32)
        if self._embeddings is None:
            self._embeddings = vec
        else:
            self._embeddings = np.vstack([self._embeddings, vec])
        self._paths.append(path)
        self._products.append(product)

    def add_batch(
        self,
        embeddings: np.ndarray,
        paths: list[str],
        products: list[str],
    ) -> None:
        vecs = embeddings.astype(np.float32)
        if self._embeddings is None:
            self._embeddings = vecs
        else:
            self._embeddings = np.vstack([self._embeddings, vecs])
        self._paths.extend(paths)
        self._products.extend(products)

    # ── search ─────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
        product_filter: str | None = None,
    ) -> list[ReferenceHit]:
        """Find the top-k most similar references to *query*."""
        if self._embeddings is None or len(self._paths) == 0:
            return []

        q = query.flatten().astype(np.float32)

        if self._metric == "cosine":
            sims = (self._embeddings @ q.reshape(-1, 1)).squeeze(-1)
        else:
            dists = np.linalg.norm(self._embeddings - q, axis=1)
            sims = 1.0 / (1.0 + dists)

        # Optional product filter
        if product_filter:
            mask = np.array([p == product_filter for p in self._products])
            sims = np.where(mask, sims, -1.0)

        order = np.argsort(sims)[::-1][:top_k]
        hits = []
        for rank, idx in enumerate(order, 1):
            s = float(sims[idx])
            if s < min_similarity:
                continue
            hits.append(ReferenceHit(
                path=self._paths[idx],
                product=self._products[idx],
                score=s,
                rank=rank,
            ))
        return hits

    # ── persistence ────────────────────────────────────────

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            embeddings=self._embeddings if self._embeddings is not None else np.empty((0, 0)),
            paths=np.array(self._paths, dtype=object),
            products=np.array(self._products, dtype=object),
        )
        logger.info("Reference store saved → %s  (%d entries)", p, len(self))

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self._embeddings = data["embeddings"].astype(np.float32)
        self._paths = list(data["paths"])
        self._products = list(data["products"])
        logger.info("Reference store loaded ← %s  (%d entries)", path, len(self))

    def summary(self) -> str:
        n = len(self)
        products = set(self._products)
        return f"{n} references, {len(products)} products"
