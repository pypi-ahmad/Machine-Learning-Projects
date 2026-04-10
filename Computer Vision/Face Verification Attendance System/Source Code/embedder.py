"""Embedding extraction module for Face Verification Attendance System.

Uses InsightFace ArcFace (buffalo_l) to extract 512-d normalized
face embeddings from detected face crops or full frames.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceAttendanceConfig

log = logging.getLogger("face_attendance.embedder")


class FaceEmbedder:
    """InsightFace ArcFace embedding extractor.

    Wraps ``insightface.app.FaceAnalysis`` for normalized face embeddings.
    Returns the shared FaceAnalysis instance so the detector can reuse it.
    """

    def __init__(self, cfg: FaceAttendanceConfig) -> None:
        self.cfg = cfg
        self._app = None
        self._ready = False

    def load(self):
        """Initialize InsightFace FaceAnalysis.

        Returns
        -------
        FaceAnalysis or None
            The initialized app (also usable for detection), or None on error.
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
        """Underlying FaceAnalysis instance (for detector sharing)."""
        return self._app

    @property
    def ready(self) -> bool:
        return self._ready

    def extract(self, frame: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for all faces detected in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        list[np.ndarray]
            List of normalized 512-d embedding vectors.
        """
        if not self._ready:
            return []
        faces = self._app.get(frame)
        embeddings = []
        for face in faces:
            if face.normed_embedding is not None:
                embeddings.append(face.normed_embedding)
        return embeddings

    def extract_single(self, crop: np.ndarray) -> np.ndarray | None:
        """Extract embedding from a single face crop.

        Runs InsightFace on the crop and returns the embedding of the
        largest detected face, or None if no face is found.

        Parameters
        ----------
        crop : np.ndarray
            BGR face crop.

        Returns
        -------
        np.ndarray or None
            Normalized 512-d embedding, or None.
        """
        if not self._ready:
            return None
        faces = self._app.get(crop)
        if not faces:
            return None
        # Pick largest face in the crop
        best = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return best.normed_embedding

    def extract_from_frame(
        self, frame: np.ndarray,
    ) -> list[dict]:
        """Extract face info with bboxes + embeddings from a full frame.

        Returns
        -------
        list[dict]
            Each dict has ``box``, ``confidence``, ``embedding``.
        """
        if not self._ready:
            return []
        faces = self._app.get(frame)
        results = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            entry = {
                "box": (x1, y1, x2, y2),
                "confidence": float(face.det_score),
                "embedding": face.normed_embedding,
            }
            results.append(entry)
        return results
