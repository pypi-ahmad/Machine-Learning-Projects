"""Embedding extraction module for Face Verification Attendance System.

Uses InsightFace ArcFace (buffalo_l) to extract 512-d normalized
face embeddings from detected face crops or full frames.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceAttendanceConfig

log = logging.getLogger("face_attendance.embedder")
MIN_INFERENCE_SIDE = 224
MIN_CONTEXT_SIDE = 160
CONTEXT_PAD_RATIO = 0.65
CONTEXT_PAD_MIN = 48


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
                "pip install insightface onnxruntime"
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

    def _prepare_input(self, frame: np.ndarray) -> tuple[np.ndarray, float, int]:
        height, width = frame.shape[:2]
        pad = 0
        min_side = min(height, width)
        if min_side < MIN_CONTEXT_SIDE:
            pad = max(CONTEXT_PAD_MIN, int(round(min_side * CONTEXT_PAD_RATIO)))
            frame = cv2.copyMakeBorder(
                frame,
                pad,
                pad,
                pad,
                pad,
                cv2.BORDER_REFLECT_101,
            )
            height, width = frame.shape[:2]
            min_side = min(height, width)

        if min_side >= MIN_INFERENCE_SIDE:
            return frame, 1.0, pad

        scale = MIN_INFERENCE_SIDE / float(min_side)
        resized = cv2.resize(
            frame,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        return resized, scale, pad

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
        prepared_frame, _scale, _pad = self._prepare_input(frame)
        faces = self._app.get(prepared_frame)
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
        prepared_crop, _scale, _pad = self._prepare_input(crop)
        faces = self._app.get(prepared_crop)
        if not faces:
            return None
        # Pick largest face in the crop
        best = max(
            faces,
            key=lambda f: (
                float(getattr(f, "det_score", 0.0)),
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            ),
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
        prepared_frame, scale, pad = self._prepare_input(frame)
        faces = self._app.get(prepared_frame)
        height, width = frame.shape[:2]
        results = []
        for face in faces:
            raw_x1, raw_y1, raw_x2, raw_y2 = face.bbox / scale
            preclip_x1 = raw_x1 - pad
            preclip_y1 = raw_y1 - pad
            preclip_x2 = raw_x2 - pad
            preclip_y2 = raw_y2 - pad
            preclip_area = max(0.0, preclip_x2 - preclip_x1) * max(0.0, preclip_y2 - preclip_y1)
            x1 = max(0, int(round(preclip_x1)))
            y1 = max(0, int(round(preclip_y1)))
            x2 = min(width, int(round(preclip_x2)))
            y2 = min(height, int(round(preclip_y2)))
            clipped_area = max(0, x2 - x1) * max(0, y2 - y1)
            if clipped_area == 0:
                continue
            if preclip_area > 0 and (clipped_area / preclip_area) < 0.6:
                continue
            entry = {
                "box": (x1, y1, x2, y2),
                "confidence": float(face.det_score),
                "embedding": face.normed_embedding,
            }
            results.append(entry)
        return results
