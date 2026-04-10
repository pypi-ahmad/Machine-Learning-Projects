"""Face detection module for Face Verification Attendance System.

Supports two backends:
1. YOLO face detector (custom fine-tuned on face data) — preferred
2. InsightFace built-in detector — fallback

Returns bounding boxes and face crops for downstream embedding extraction.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceAttendanceConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("face_attendance.detector")


@dataclass
class DetectedFace:
    """Single detected face with bounding box and crop."""

    box: tuple[int, int, int, int]      # (x1, y1, x2, y2)
    crop: np.ndarray                     # BGR face crop
    confidence: float


class FaceDetector:
    """Multi-backend face detector.

    Resolution:
    1. YOLO face detector (``weights/face_detect_yolo26m.pt``)
    2. InsightFace built-in ``RetinaFace`` (via FaceAnalysis)
    """

    def __init__(self, cfg: FaceAttendanceConfig) -> None:
        self.cfg = cfg
        self._yolo = None
        self._insightface = None
        self._backend: str = "none"

    def load(self, insightface_app=None) -> str:
        """Initialize detector backend.

        Parameters
        ----------
        insightface_app : FaceAnalysis, optional
            Pre-initialized InsightFace app (shared with embedder).

        Returns
        -------
        str
            Backend name: ``"yolo_face"``, ``"insightface"``, or ``"none"``.
        """
        # Try YOLO face detector first
        if self.cfg.use_yolo_detector:
            try:
                from models.registry import resolve
                from utils.yolo import load_yolo

                weights, _ver, _fb = resolve(
                    "face_verification_attendance", "face_detect",
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

        # Fallback: InsightFace detector (shared instance)
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
        """Detect faces and return bounding boxes + crops.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        list[DetectedFace]
            Detected faces sorted by area (largest first).
        """
        if self._backend == "yolo_face":
            return self._detect_yolo(frame)
        if self._backend == "insightface":
            return self._detect_insightface(frame)
        return []

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
        # Sort largest first
        faces.sort(
            key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]),
            reverse=True,
        )
        return faces

    def _detect_insightface(self, frame: np.ndarray) -> list[DetectedFace]:
        results = self._insightface.get(frame)
        h, w = frame.shape[:2]
        faces: list[DetectedFace] = []
        for face in results:
            x1, y1, x2, y2 = map(int, face.bbox)
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
            key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]),
            reverse=True,
        )
        return faces
