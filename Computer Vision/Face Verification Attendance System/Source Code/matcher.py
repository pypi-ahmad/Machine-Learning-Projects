"""Face matching module for Face Verification Attendance System.

Performs threshold-based cosine similarity matching of query embeddings
against the enrolled gallery.  Handles unknown-face labelling.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceAttendanceConfig

log = logging.getLogger("face_attendance.matcher")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class MatchResult:
    """Result of matching one face against the gallery."""

    identity: str                   # matched name or "Unknown"
    similarity: float               # best cosine similarity
    matched: bool                   # above threshold?
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    det_confidence: float           # detection confidence


class FaceMatcher:
    """Compare query embeddings against an enrolled gallery."""

    def __init__(self, cfg: FaceAttendanceConfig) -> None:
        self.cfg = cfg
        self._gallery: dict[str, np.ndarray] = {}

    def set_gallery(self, gallery: dict[str, np.ndarray]) -> None:
        """Load the gallery of enrolled identities."""
        self._gallery = gallery

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    def match(
        self,
        embedding: np.ndarray,
        box: tuple[int, int, int, int] = (0, 0, 0, 0),
        det_confidence: float = 0.0,
    ) -> MatchResult:
        """Match a single embedding against the gallery.

        Parameters
        ----------
        embedding : np.ndarray
            512-d normalized face embedding.
        box : tuple
            Bounding box (x1, y1, x2, y2).
        det_confidence : float
            Detection confidence.

        Returns
        -------
        MatchResult
            Best match result.
        """
        if not self._gallery:
            return MatchResult(
                identity=self.cfg.unknown_label,
                similarity=0.0,
                matched=False,
                box=box,
                det_confidence=det_confidence,
            )

        best_name = self.cfg.unknown_label
        best_score = 0.0

        for name, gallery_emb in self._gallery.items():
            score = cosine_similarity(embedding, gallery_emb)
            if score > best_score:
                best_score = score
                best_name = name

        matched = best_score >= self.cfg.similarity_threshold
        identity = best_name if matched else self.cfg.unknown_label

        return MatchResult(
            identity=identity,
            similarity=best_score,
            matched=matched,
            box=box,
            det_confidence=det_confidence,
        )

    def match_batch(
        self,
        face_data: list[dict],
    ) -> list[MatchResult]:
        """Match a batch of detected faces.

        Parameters
        ----------
        face_data : list[dict]
            Each dict must have ``embedding``, ``box``, ``confidence``.

        Returns
        -------
        list[MatchResult]
        """
        results = []
        for fd in face_data:
            result = self.match(
                embedding=fd["embedding"],
                box=fd.get("box", (0, 0, 0, 0)),
                det_confidence=fd.get("confidence", 0.0),
            )
            results.append(result)
        return results
