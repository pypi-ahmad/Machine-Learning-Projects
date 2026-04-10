"""Facial landmark detection module for Driver Drowsiness Monitor.

Wraps MediaPipe Face Mesh to extract 468 dense face landmarks
per frame. Provides normalized and pixel-space landmark accessors.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DrowsinessConfig

log = logging.getLogger("drowsiness.landmark_detector")

# ── MediaPipe Face Mesh landmark indices ──────────────────
# 6-point eye contour for Eye Aspect Ratio (EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Inner lip contour for Mouth Aspect Ratio (MAR)
UPPER_LIP = [13]       # top inner lip
LOWER_LIP = [14]       # bottom inner lip
LEFT_MOUTH = [78]      # left corner
RIGHT_MOUTH = [308]    # right corner
# Extended vertical: upper-inner [13, 312, 311, 310] / lower-inner [14, 317, 402, 318]
MOUTH_VERTICAL_TOP = [13, 312, 311]
MOUTH_VERTICAL_BOT = [14, 317, 402]
MOUTH_HORIZONTAL = [78, 308]

# Nose + chin + forehead proxies for head pose
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER = 33
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# 3D model points for solvePnP (generic face model, mm)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye corner
    (225.0, 170.0, -135.0),    # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)


@dataclass
class LandmarkResult:
    """Result from landmark detection on one frame."""

    detected: bool = False
    landmarks: list = field(default_factory=list)  # MediaPipe NormalizedLandmarkList
    frame_h: int = 0
    frame_w: int = 0

    def pixel_coords(self, index: int) -> tuple[float, float]:
        """Get (x, y) in pixel space for landmark *index*."""
        lm = self.landmarks[index]
        return lm.x * self.frame_w, lm.y * self.frame_h

    def pixel_coords_3d(self, index: int) -> tuple[float, float, float]:
        """Get (x, y, z) — z is depth estimate from MediaPipe."""
        lm = self.landmarks[index]
        return lm.x * self.frame_w, lm.y * self.frame_h, lm.z * self.frame_w


class LandmarkDetector:
    """MediaPipe Face Mesh landmark detector."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self._face_mesh = None
        self._ready = False

    def load(self) -> bool:
        """Initialize MediaPipe Face Mesh.

        Returns
        -------
        bool
            True if loaded successfully.
        """
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
            self._mp_styles = mp.solutions.drawing_styles
            self._mp_face_mesh = mp.solutions.face_mesh
            self._ready = True
            log.info("MediaPipe Face Mesh loaded (468 landmarks)")
            return True
        except ImportError:
            log.error("MediaPipe not installed. pip install mediapipe")
        except Exception as exc:
            log.error("MediaPipe init failed: %s", exc)
        return False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def mp_draw(self):
        return self._mp_draw

    @property
    def mp_styles(self):
        return self._mp_styles

    @property
    def mp_face_mesh(self):
        return self._mp_face_mesh

    def detect(self, frame: np.ndarray) -> LandmarkResult:
        """Detect face landmarks in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        LandmarkResult
        """
        if not self._ready:
            return LandmarkResult()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return LandmarkResult(frame_h=h, frame_w=w)

        # Use first face (driver)
        face_lms = results.multi_face_landmarks[0]

        return LandmarkResult(
            detected=True,
            landmarks=face_lms.landmark,
            frame_h=h,
            frame_w=w,
        )
