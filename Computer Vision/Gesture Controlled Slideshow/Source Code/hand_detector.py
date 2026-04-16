"""MediaPipe Hand Landmarker wrapper for Gesture Controlled Slideshow.

Detects hand landmarks (21 per hand) via MediaPipe Hand Landmarker.

Landmark indices (per hand):
    0  WRIST
    1  THUMB_CMC      5  INDEX_MCP     9  MIDDLE_MCP
    2  THUMB_MCP      6  INDEX_PIP    10  MIDDLE_PIP
    3  THUMB_IP       7  INDEX_DIP    11  MIDDLE_DIP
    4  THUMB_TIP      8  INDEX_TIP    12  MIDDLE_TIP
   13  RING_MCP      17  PINKY_MCP
   14  RING_PIP      18  PINKY_PIP
   15  RING_DIP      19  PINKY_DIP
   16  RING_TIP      20  PINKY_TIP
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
from config import GestureConfig

log = logging.getLogger("gesture.hand_detector")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Finger tip and PIP joint indices for finger-up detection
FINGER_TIPS = [8, 12, 16, 20]      # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
WRIST = 0


@dataclass
class HandResult:
    """Result from hand detection on one frame."""

    detected: bool = False
    landmarks: list = field(default_factory=list)   # MediaPipe landmark list
    handedness: str = ""                             # "Left" or "Right"
    score: float = 0.0
    frame_h: int = 0
    frame_w: int = 0

    def pixel_coords(self, index: int) -> tuple[float, float]:
        lm = self.landmarks[index]
        return lm.x * self.frame_w, lm.y * self.frame_h

    def normalized(self, index: int) -> tuple[float, float, float]:
        lm = self.landmarks[index]
        return lm.x, lm.y, lm.z


class HandDetector:
    """MediaPipe Hand Landmarker wrapper."""

    def __init__(self, cfg: GestureConfig) -> None:
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
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path),
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=self.cfg.max_num_hands,
                min_hand_detection_confidence=self.cfg.min_detection_confidence,
                min_hand_presence_confidence=self.cfg.min_presence_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._mp = mp
            self._ready = True
            log.info("MediaPipe Hand Landmarker loaded: %s", model_path.name)
            return True
        except ImportError:
            log.error("MediaPipe not installed. pip install mediapipe")
        except Exception as exc:
            log.error("MediaPipe init failed: %s", exc)
        return False

    @property
    def ready(self) -> bool:
        return self._ready

    def detect(self, frame: np.ndarray) -> HandResult:
        """Detect hand landmarks in a BGR frame.

        Returns the first detected hand.
        """
        if not self._ready:
            return HandResult()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        results = self._landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return HandResult(frame_h=h, frame_w=w)

        hand_lms = results.hand_landmarks[0]

        # Handedness
        handedness = ""
        score = 0.0
        if results.handedness:
            cls = results.handedness[0][0]
            handedness = cls.category_name
            score = cls.score

        return HandResult(
            detected=True,
            landmarks=list(hand_lms),
            handedness=handedness,
            score=score,
            frame_h=h,
            frame_w=w,
        )

    def draw_landmarks(self, frame: np.ndarray, hand: HandResult) -> None:
        """Draw hand landmarks on frame (in-place)."""
        if not hand.detected or not self._ready:
            return
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand.pixel_coords(start_idx)
            end = hand.pixel_coords(end_idx)
            cv2.line(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                (0, 255, 255),
                2,
            )

        for idx in range(len(hand.landmarks)):
            px, py = hand.pixel_coords(idx)
            cv2.circle(frame, (int(px), int(py)), 3, (255, 100, 0), -1)


def _ensure_landmarker_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".download")
    with urllib.request.urlopen(MODEL_URL) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(MODEL_PATH)
    return MODEL_PATH
