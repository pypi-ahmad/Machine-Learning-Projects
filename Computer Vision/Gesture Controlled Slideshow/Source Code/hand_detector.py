"""MediaPipe Hand Landmarker wrapper for Gesture Controlled Slideshow.

Detects hand landmarks (21 per hand) via MediaPipe Hands.

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
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GestureConfig

log = logging.getLogger("gesture.hand_detector")

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
    """MediaPipe Hands detector."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._hands = None
        self._ready = False

    def load(self) -> bool:
        try:
            import mediapipe as mp

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.cfg.max_num_hands,
                model_complexity=self.cfg.model_complexity,
                min_detection_confidence=self.cfg.min_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
            )
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_styles = mp.solutions.drawing_styles
            self._ready = True
            log.info("MediaPipe Hands loaded (21 landmarks)")
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
    def mp_hands(self):
        return self._mp_hands

    @property
    def mp_draw(self):
        return self._mp_draw

    @property
    def mp_styles(self):
        return self._mp_styles

    def detect(self, frame: np.ndarray) -> HandResult:
        """Detect hand landmarks in a BGR frame.

        Returns the first detected hand.
        """
        if not self._ready:
            return HandResult()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return HandResult(frame_h=h, frame_w=w)

        hand_lms = results.multi_hand_landmarks[0]

        # Handedness
        handedness = ""
        score = 0.0
        if results.multi_handedness:
            cls = results.multi_handedness[0].classification[0]
            handedness = cls.label
            score = cls.score

        return HandResult(
            detected=True,
            landmarks=hand_lms.landmark,
            handedness=handedness,
            score=score,
            frame_h=h,
            frame_w=w,
        )

    def draw_landmarks(self, frame: np.ndarray, hand: HandResult) -> None:
        """Draw hand landmarks on frame (in-place)."""
        if not hand.detected or not self._ready:
            return
        import mediapipe as mp

        hand_lms = mp.solutions.hands.HandLandmark
        # Reconstruct NormalizedLandmarkList for draw utility
        landmark_list = type(
            "Obj", (), {"landmark": hand.landmarks},
        )()
        self._mp_draw.draw_landmarks(
            frame,
            landmark_list,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_styles.get_default_hand_landmarks_style(),
            self._mp_styles.get_default_hand_connections_style(),
        )
