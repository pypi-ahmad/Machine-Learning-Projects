"""Face detection module for Face Clustering Photo Organizer.

Supports two backends:
1. YOLO face detector (custom fine-tuned) — preferred
2. InsightFace built-in detector — fallback

Returns bounding boxes and face crops for downstream embedding.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceClusterConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("face_cluster.detector")
MIN_INFERENCE_SIDE = 224
MIN_CONTEXT_SIDE = 160
CONTEXT_PAD_RATIO = 0.65
CONTEXT_PAD_MIN = 48


@dataclass
class DetectedFace:
    """Single detected face."""

    box: tuple[int, int, int, int]      # (x1, y1, x2, y2)
    crop: np.ndarray                     # BGR face crop
    confidence: float


class FaceDetector:
    """Multi-backend face detector."""

    def __init__(self, cfg: FaceClusterConfig) -> None:
        self.cfg = cfg
        self._yolo = None
        self._insightface = None
        self._backend: str = "none"

    def load(self, insightface_app=None) -> str:
        """Initialize detector.

        Parameters
        ----------
        insightface_app : FaceAnalysis, optional
            Pre-initialized InsightFace app (shared with embedder).

        Returns
        -------
        str
            Backend name used.
        """
        if self.cfg.use_yolo_detector:
            try:
                from models.registry import resolve
                from utils.yolo import load_yolo

                weights, _ver, _fb = resolve(
                    "face_clustering_photo_organizer", "face_detect",
                )
                w_path = (
                    Path(weights)
                    if Path(weights).is_absolute()
                    else REPO_ROOT / weights
                )
                if w_path.exists():
                    self._yolo = load_yolo(str(w_path))
                    self._backend = "yolo_face"
                    log.info("YOLO face detector loaded: %s", w_path.name)
                    return self._backend
            except Exception as exc:
                log.debug("YOLO face detector unavailable: %s", exc)

        if insightface_app is not None:
            self._insightface = insightface_app
            self._backend = "insightface"
            log.info("Using InsightFace built-in detector")
            return self._backend

        log.warning("No face detector available")
        return self._backend

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces in a BGR image."""
        if self._backend == "yolo_face":
            return self._detect_yolo(frame)
        if self._backend == "insightface":
            return self._detect_insightface(frame)
        return []

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

    def _detect_yolo(self, frame: np.ndarray) -> list[DetectedFace]:
        results = self._yolo(
            frame, verbose=False, conf=self.cfg.det_confidence,
        )
        h, w = frame.shape[:2]
        faces: list[DetectedFace] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            faces.append(DetectedFace(
                box=(x1, y1, x2, y2),
                crop=crop,
                confidence=float(box.conf[0]),
            ))
        faces.sort(
            key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]),
            reverse=True,
        )
        return faces

    def _detect_insightface(self, frame: np.ndarray) -> list[DetectedFace]:
        prepared_frame, scale, pad = self._prepare_input(frame)
        results = self._insightface.get(prepared_frame)
        h, w = frame.shape[:2]
        faces: list[DetectedFace] = []
        for face in results:
            raw_x1, raw_y1, raw_x2, raw_y2 = face.bbox / scale
            preclip_x1 = raw_x1 - pad
            preclip_y1 = raw_y1 - pad
            preclip_x2 = raw_x2 - pad
            preclip_y2 = raw_y2 - pad
            preclip_area = max(0.0, preclip_x2 - preclip_x1) * max(0.0, preclip_y2 - preclip_y1)
            x1 = max(0, int(round(preclip_x1)))
            y1 = max(0, int(round(preclip_y1)))
            x2 = min(w, int(round(preclip_x2)))
            y2 = min(h, int(round(preclip_y2)))
            clipped_area = max(0, x2 - x1) * max(0, y2 - y1)
            if clipped_area == 0:
                continue
            if preclip_area > 0 and (clipped_area / preclip_area) < 0.6:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            faces.append(DetectedFace(
                box=(x1, y1, x2, y2),
                crop=crop,
                confidence=float(face.det_score),
            ))
        faces.sort(
            key=lambda f: (
                f.confidence,
                (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]),
            ),
            reverse=True,
        )
        return faces
