"""Sign Language Alphabet Recognizer -- MediaPipe Hand Landmarker wrapper."""

from __future__ import annotations

import dataclasses
import shutil
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

WRIST = 0
NUM_LANDMARKS = 21

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


@dataclasses.dataclass
class HandResult:
    """Detection result for a single hand."""

    detected: bool
    landmarks: list | None  # NormalizedLandmark list (len 21)
    handedness: str  # "Left" or "Right"
    score: float
    frame_h: int
    frame_w: int

    def pixel(self, idx: int) -> tuple[int, int]:
        lm = self.landmarks[idx]
        return int(lm.x * self.frame_w), int(lm.y * self.frame_h)

    def norm(self, idx: int) -> tuple[float, float, float]:
        lm = self.landmarks[idx]
        return lm.x, lm.y, lm.z


class HandDetector:
    """Lazy-loading MediaPipe Hand Landmarker wrapper."""

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._max = max_num_hands
        self._complexity = model_complexity
        self._det_conf = min_detection_confidence
        self._presence_conf = min_presence_confidence
        self._trk_conf = min_tracking_confidence
        self._static = static_image_mode
        self._landmarker = None
        self._mp = None

    def load(self) -> None:
        import mediapipe as mp

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_landmarker_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=self._max,
            min_hand_detection_confidence=self._det_conf,
            min_hand_presence_confidence=self._presence_conf,
            min_tracking_confidence=self._trk_conf,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    @property
    def ready(self) -> bool:
        return self._landmarker is not None

    def detect(self, frame: np.ndarray) -> HandResult | None:
        """Return the first detected hand, or *None*."""
        return self._detect_impl(frame)

    def detect_for_image(self, frame: np.ndarray) -> HandResult | None:
        """Return the first detected hand in a static image."""
        return self._detect_impl(frame)

    def _detect_impl(self, frame: np.ndarray) -> HandResult | None:
        import cv2

        if not self.ready:
            self.load()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        res = self._landmarker.detect(mp_image)

        if res.hand_landmarks and res.handedness:
            lm = res.hand_landmarks[0]
            hd = res.handedness[0][0]
            return HandResult(
                detected=True,
                landmarks=list(lm),
                handedness=hd.category_name,
                score=hd.score,
                frame_h=h,
                frame_w=w,
            )
        return None

    def draw_landmarks(self, frame: np.ndarray, hand: HandResult) -> np.ndarray:
        """Draw hand skeleton on *frame* (in-place)."""
        import cv2

        if hand.landmarks is None:
            return frame
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, hand.pixel(start_idx), hand.pixel(end_idx), (0, 255, 0), 2)
        for idx in range(len(hand.landmarks)):
            cv2.circle(frame, hand.pixel(idx), 3, (255, 255, 255), -1)
        return frame

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def _ensure_landmarker_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".download")
    with urllib.request.urlopen(MODEL_URL) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(MODEL_PATH)
    return MODEL_PATH
