"""Sign Language Alphabet Recognizer — MediaPipe Hands wrapper."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

WRIST = 0
NUM_LANDMARKS = 21


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
    """Lazy-loading MediaPipe Hands wrapper."""

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._max = max_num_hands
        self._complexity = model_complexity
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._static = static_image_mode
        self._hands = None
        self._mp_hands = None
        self._mp_draw = None

    def load(self) -> None:
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=self._static,
            max_num_hands=self._max,
            model_complexity=self._complexity,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )

    @property
    def ready(self) -> bool:
        return self._hands is not None

    def detect(self, frame: np.ndarray) -> HandResult | None:
        """Return the first detected hand, or *None*."""
        import cv2

        if not self.ready:
            self.load()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)

        if res.multi_hand_landmarks and res.multi_handedness:
            lm = res.multi_hand_landmarks[0]
            hd = res.multi_handedness[0]
            return HandResult(
                detected=True,
                landmarks=lm.landmark,
                handedness=hd.classification[0].label,
                score=hd.classification[0].score,
                frame_h=h,
                frame_w=w,
            )
        return None

    def detect_for_image(self, frame: np.ndarray) -> HandResult | None:
        """Detect in a static image (creates an ad-hoc static-mode Hands)."""
        import cv2
        import mediapipe as mp

        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=self._complexity,
            min_detection_confidence=self._det_conf,
        )
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        hands.close()

        if res.multi_hand_landmarks and res.multi_handedness:
            lm = res.multi_hand_landmarks[0]
            hd = res.multi_handedness[0]
            return HandResult(
                detected=True,
                landmarks=lm.landmark,
                handedness=hd.classification[0].label,
                score=hd.classification[0].score,
                frame_h=h,
                frame_w=w,
            )
        return None

    def draw_landmarks(self, frame: np.ndarray, hand: HandResult) -> np.ndarray:
        """Draw hand skeleton on *frame* (in-place)."""
        import mediapipe as mp

        if hand.landmarks is None:
            return frame
        lm_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for pt in hand.landmarks:
            lm = lm_proto.landmark.add()
            lm.x, lm.y, lm.z = pt.x, pt.y, pt.z
        self._mp_draw.draw_landmarks(
            frame,
            lm_proto,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self._mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
        )
        return frame

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
