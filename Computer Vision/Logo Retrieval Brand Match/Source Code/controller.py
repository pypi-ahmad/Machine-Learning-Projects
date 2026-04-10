"""Logo Retrieval Brand Match — high-level controller.

Orchestrates optional detection → embedding → retrieval.
Detection and retrieval are kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import LogoConfig
from retriever import LogoRetriever, RetrievalResult


@dataclass
class QueryResult:
    """Complete result for one query image."""

    image: np.ndarray
    retrieval: RetrievalResult
    detection_used: bool = False


class LogoController:
    """Top-level orchestrator: detect (optional) → embed → retrieve."""

    def __init__(self, cfg: LogoConfig | None = None) -> None:
        self.cfg = cfg or LogoConfig()
        self._retriever = LogoRetriever(self.cfg)
        self._detector = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self, index_path: str | None = None) -> None:
        """Load embedding model and index."""
        self._retriever.load(index_path)

        if self.cfg.use_detector:
            from detector import LogoDetector
            self._detector = LogoDetector(
                model_name=self.cfg.detector_model,
                confidence=self.cfg.detector_conf,
                device=self.cfg.device,
            )
            self._detector.load()

    def close(self) -> None:
        self._retriever.close()
        if self._detector is not None:
            self._detector.close()

    # ── query API ─────────────────────────────────────────

    def query(
        self,
        image_bgr: np.ndarray,
        top_k: int | None = None,
        source: str | None = None,
    ) -> QueryResult:
        """Run full pipeline: detect (optional) → embed → retrieve."""
        detection_used = False

        if self._detector is not None:
            det = self._detector.detect(image_bgr)
            if det.found:
                # Use the highest-confidence crop
                best_idx = int(np.argmax(det.confidences))
                image_bgr = det.crops[best_idx]
                detection_used = True

        retrieval = self._retriever.query_image(
            image_bgr, top_k=top_k, source=source,
        )

        return QueryResult(
            image=image_bgr,
            retrieval=retrieval,
            detection_used=detection_used,
        )

    @property
    def index(self):
        return self._retriever.index
