"""Embedding extraction module for Face Clustering Photo Organizer.

Uses InsightFace ArcFace (buffalo_l) to extract 512-d normalized
face embeddings from detected face crops or full frames.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceClusterConfig

log = logging.getLogger("face_cluster.embedder")


class FaceEmbedder:
    """InsightFace ArcFace embedding extractor."""

    def __init__(self, cfg: FaceClusterConfig) -> None:
        self.cfg = cfg
        self._app = None
        self._ready = False

    def load(self):
        """Initialize InsightFace FaceAnalysis.

        Returns
        -------
        FaceAnalysis or None
            Initialized app (shareable with detector), or None on error.
        """
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=self.cfg.embedding_model,
                providers=self.cfg.providers,
            )
            self._app.prepare(
                ctx_id=0,
                det_size=self.cfg.det_size,
            )
            self._ready = True
            log.info(
                "InsightFace loaded (model=%s, dim=%d)",
                self.cfg.embedding_model,
                self.cfg.embedding_dim,
            )
            return self._app
        except ImportError:
            log.error(
                "InsightFace not installed. "
                "pip install insightface onnxruntime-gpu"
            )
        except Exception as exc:
            log.error("InsightFace init failed: %s", exc)
        return None

    @property
    def app(self):
        return self._app

    @property
    def ready(self) -> bool:
        return self._ready

    def extract_from_frame(
        self, frame: np.ndarray,
    ) -> list[dict]:
        """Extract face bboxes + embeddings from a full frame.

        Returns
        -------
        list[dict]
            Each dict has ``box``, ``confidence``, ``embedding``, ``crop``.
        """
        if not self._ready:
            return []
        h, w = frame.shape[:2]
        faces = self._app.get(frame)
        results = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or face.normed_embedding is None:
                continue
            results.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(face.det_score),
                "embedding": face.normed_embedding,
                "crop": crop,
            })
        return results

    def extract_single(self, crop: np.ndarray) -> np.ndarray | None:
        """Extract embedding from a single face crop."""
        if not self._ready:
            return None
        faces = self._app.get(crop)
        if not faces:
            return None
        best = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return best.normed_embedding
