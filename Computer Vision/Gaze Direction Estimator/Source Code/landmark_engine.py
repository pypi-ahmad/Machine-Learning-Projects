"""MediaPipe face landmark detection engine.

Wraps MediaPipe Face Landmarker to extract dense face landmarks
per frame, including the iris indices used by the gaze heuristics.

Iris landmark indices:
    - Left iris:  468 (center), 469, 470, 471, 472
    - Right iris: 473 (center), 474, 475, 476, 477
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
from config import GazeConfig

log = logging.getLogger("gaze.landmark_engine")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


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
    """MediaPipe Face Landmarker wrapper with iris landmarks."""

    def __init__(self, cfg: GazeConfig) -> None:
        self.cfg = cfg
        self._landmarker = None
        self._mp = None
        self._ready = False

    def load(self) -> bool:
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
        """Detect face landmarks in a BGR frame."""
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
