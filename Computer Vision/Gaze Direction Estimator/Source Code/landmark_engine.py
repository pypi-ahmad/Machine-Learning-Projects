"""MediaPipe Face Mesh landmark detection engine.

Wraps MediaPipe to extract 478 face landmarks (468 base + 10 iris)
per frame.  ``refine_landmarks=True`` is required for iris indices.

Iris landmark indices (MediaPipe with refinement):
    - Left iris:  468 (center), 469, 470, 471, 472
    - Right iris: 473 (center), 474, 475, 476, 477
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GazeConfig

log = logging.getLogger("gaze.landmark_engine")


@dataclass
class LandmarkResult:
    """Result from landmark detection on one frame."""

    detected: bool = False
    landmarks: list = field(default_factory=list)
    frame_h: int = 0
    frame_w: int = 0

    def pixel_coords(self, index: int) -> tuple[float, float]:
        lm = self.landmarks[index]
        return lm.x * self.frame_w, lm.y * self.frame_h


class LandmarkEngine:
    """MediaPipe Face Mesh detector with iris refinement."""

    def __init__(self, cfg: GazeConfig) -> None:
        self.cfg = cfg
        self._face_mesh = None
        self._ready = False

    def load(self) -> bool:
        try:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.cfg.max_num_faces,
                refine_landmarks=self.cfg.refine_landmarks,
                min_detection_confidence=self.cfg.min_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_face_mesh = mp.solutions.face_mesh
            self._ready = True
            log.info("MediaPipe Face Mesh loaded (478 landmarks with iris)")
            return True
        except ImportError:
            log.error("MediaPipe not installed. pip install mediapipe")
        except Exception as exc:
            log.error("MediaPipe init failed: %s", exc)
        return False

    @property
    def ready(self) -> bool:
        return self._ready

    def detect(self, frame: np.ndarray) -> LandmarkResult:
        """Detect face landmarks in a BGR frame."""
        if not self._ready:
            return LandmarkResult()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return LandmarkResult(frame_h=h, frame_w=w)

        face_lms = results.multi_face_landmarks[0]
        return LandmarkResult(
            detected=True,
            landmarks=face_lms.landmark,
            frame_h=h,
            frame_w=w,
        )
