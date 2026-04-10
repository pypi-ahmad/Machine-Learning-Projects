"""Product Counterfeit Visual Checker — comparison engine.

Compares a suspect product image against approved references using:
1. Global embedding similarity (cosine)
2. Region-patch embedding similarity (grid-based)
3. Colour-histogram similarity

Produces a composite mismatch-risk score —  **screening only**,
not a definitive counterfeit determination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import CounterfeitConfig
from embedder import ProductEmbedder
from reference_store import ReferenceHit, ReferenceStore

logger = logging.getLogger(__name__)


@dataclass
class ComparisonDetail:
    """Breakdown of one suspect ↔ reference comparison."""

    reference_path: str
    reference_product: str
    global_score: float
    region_score: float
    histogram_score: float
    composite_score: float
    region_scores: list[float] = field(default_factory=list)


@dataclass
class ScreeningResult:
    """Full screening result for one suspect image."""

    suspect_path: str | None
    risk_level: str                         # "low" | "medium" | "high"
    best_composite: float                   # highest composite vs. any ref
    best_reference: str | None              # path of best-matching reference
    best_product: str | None
    details: list[ComparisonDetail]         # per-reference breakdown
    reference_hits: list[ReferenceHit]      # raw retrieval hits

    @property
    def mismatch_risk_pct(self) -> float:
        """Mismatch risk as a percentage (higher = more suspicious)."""
        return round((1.0 - self.best_composite) * 100, 1)


class ProductComparator:
    """Compare suspect images against approved references."""

    def __init__(
        self,
        embedder: ProductEmbedder,
        store: ReferenceStore,
        cfg: CounterfeitConfig | None = None,
    ) -> None:
        self.cfg = cfg or CounterfeitConfig()
        self._embedder = embedder
        self._store = store

    # ── main comparison API ────────────────────────────────

    def screen(
        self,
        suspect_bgr: np.ndarray,
        suspect_path: str | None = None,
        product_filter: str | None = None,
    ) -> ScreeningResult:
        """Screen a suspect image against the reference store.

        Returns a ScreeningResult with risk level and comparison details.
        """
        # 1. Global embedding
        suspect_emb = self._embedder.embed(suspect_bgr)

        # 2. Find closest references
        hits = self._store.search(
            suspect_emb,
            top_k=self.cfg.top_k,
            min_similarity=self.cfg.min_similarity,
            product_filter=product_filter,
        )

        if not hits:
            return ScreeningResult(
                suspect_path=suspect_path,
                risk_level="high",
                best_composite=0.0,
                best_reference=None,
                best_product=None,
                details=[],
                reference_hits=[],
            )

        # 3. Detailed comparison against each hit
        details: list[ComparisonDetail] = []
        for hit in hits:
            ref_img = cv2.imread(hit.path)
            if ref_img is None:
                details.append(ComparisonDetail(
                    reference_path=hit.path,
                    reference_product=hit.product,
                    global_score=hit.score,
                    region_score=0.0,
                    histogram_score=0.0,
                    composite_score=hit.score * self.cfg.global_weight,
                ))
                continue

            global_s = hit.score
            region_s, region_scores = self._region_similarity(suspect_bgr, ref_img)
            hist_s = self._histogram_similarity(suspect_bgr, ref_img)

            composite = (
                self.cfg.global_weight * global_s
                + self.cfg.region_weight * region_s
                + self.cfg.histogram_weight * hist_s
            )

            details.append(ComparisonDetail(
                reference_path=hit.path,
                reference_product=hit.product,
                global_score=global_s,
                region_score=region_s,
                histogram_score=hist_s,
                composite_score=composite,
                region_scores=region_scores,
            ))

        best = max(details, key=lambda d: d.composite_score)
        risk = self._classify_risk(best.composite_score)

        return ScreeningResult(
            suspect_path=suspect_path,
            risk_level=risk,
            best_composite=best.composite_score,
            best_reference=best.reference_path,
            best_product=best.reference_product,
            details=details,
            reference_hits=hits,
        )

    # ── region-aware comparison ────────────────────────────

    def _region_similarity(
        self,
        suspect: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[float, list[float]]:
        """Compare image patches on a grid and return mean + per-patch scores."""
        grid = self.cfg.region_grid
        suspect_patches = self._embedder.embed_patches(suspect, grid)
        ref_patches = self._embedder.embed_patches(reference, grid)

        # Cosine similarity per patch
        scores = []
        for sp, rp in zip(suspect_patches, ref_patches):
            sim = float(np.dot(sp, rp))
            scores.append(max(0.0, sim))

        mean_score = float(np.mean(scores)) if scores else 0.0
        return mean_score, scores

    # ── colour-histogram comparison ────────────────────────

    def _histogram_similarity(
        self,
        suspect: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """Compare colour histograms (HSV) using correlation."""
        bins = self.cfg.histogram_bins

        s_hsv = cv2.cvtColor(suspect, cv2.COLOR_BGR2HSV)
        r_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)

        score = 0.0
        for ch in range(3):
            s_hist = cv2.calcHist([s_hsv], [ch], None, [bins], [0, 256])
            r_hist = cv2.calcHist([r_hsv], [ch], None, [bins], [0, 256])
            cv2.normalize(s_hist, s_hist)
            cv2.normalize(r_hist, r_hist)
            score += cv2.compareHist(s_hist, r_hist, cv2.HISTCMP_CORREL)

        return max(0.0, score / 3.0)

    # ── risk classification ────────────────────────────────

    def _classify_risk(self, composite: float) -> str:
        """Classify mismatch risk from composite score."""
        if composite < self.cfg.high_risk_threshold:
            return "high"
        elif composite < self.cfg.medium_risk_threshold:
            return "medium"
        return "low"
