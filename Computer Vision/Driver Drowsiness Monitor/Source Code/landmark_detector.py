"""Facial landmark detection module for Driver Drowsiness Monitor.

Wraps MediaPipe Face Landmarker to extract dense face landmarks
per frame. Provides normalized and pixel-space landmark accessors.
"""

from __future__ import annotations

import logging
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DrowsinessConfig

log = logging.getLogger("drowsiness.landmark_detector")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

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
        """Get (x, y, z) -- z is depth estimate from MediaPipe."""
        lm = self.landmarks[index]
        return lm.x * self.frame_w, lm.y * self.frame_h, lm.z * self.frame_w


class LandmarkDetector:
    """MediaPipe Face Landmarker wrapper."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self._landmarker = None
        self._mp = None
        self._ready = False

    def load(self) -> bool:
        """Initialize MediaPipe Face Landmarker.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_landmarker_model()
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path),
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=self.cfg.max_num_faces,
                min_face_detection_confidence=self.cfg.min_detection_confidence,
                min_face_presence_confidence=self.cfg.min_presence_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self._ready = True
            log.info("MediaPipe Face Landmarker loaded: %s", model_path.name)
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
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return LandmarkResult(frame_h=h, frame_w=w)

        # Use first face (driver)
        face_lms = results.face_landmarks[0]

        return LandmarkResult(
            detected=True,
            landmarks=list(face_lms),
            frame_h=h,
            frame_w=w,
        )


def _ensure_landmarker_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".download")
    with urllib.request.urlopen(MODEL_URL) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(MODEL_PATH)
    return MODEL_PATH
