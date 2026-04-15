"""Finger Counter Pro -- MediaPipe Hands wrapper with multi-hand support."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Landmark indices
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
WRIST = 0


@dataclasses.dataclass
class HandResult:
    """Detection result for a single hand."""

    detected: bool
    landmarks: list | None  # list of NormalizedLandmark
    handedness: str  # "Left" or "Right"
    score: float
    frame_h: int
    frame_w: int

    def pixel(self, idx: int) -> tuple[int, int]:
        """Return (x, y) pixel coords for landmark *idx*."""
        lm = self.landmarks[idx]
        return int(lm.x * self.frame_w), int(lm.y * self.frame_h)

    def norm(self, idx: int) -> tuple[float, float, float]:
        """Return (x, y, z) normalised coords."""
        lm = self.landmarks[idx]
        return lm.x, lm.y, lm.z


@dataclasses.dataclass
class MultiHandResult:
    """Aggregated result for all detected hands in a frame."""

    hands: list[HandResult]
    frame_h: int
    frame_w: int

    @property
    def count(self) -> int:
        return len(self.hands)


class HandDetector:
    """Lazy-loading MediaPipe Hands wrapper (supports up to 2 hands)."""

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._max = max_num_hands
        self._complexity = model_complexity
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._hands = None
        self._mp_hands = None
        self._mp_draw = None

    def load(self) -> None:
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self._max,
            model_complexity=self._complexity,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )

    @property
    def ready(self) -> bool:
        return self._hands is not None

    def detect(self, frame: np.ndarray) -> MultiHandResult:
        """Detect hands and return :class:`MultiHandResult`."""
        import cv2

        if not self.ready:
            self.load()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)

        hands: list[HandResult] = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, hd in zip(
                res.multi_hand_landmarks, res.multi_handedness
            ):
                label = hd.classification[0].label  # "Left" / "Right"
                score = hd.classification[0].score
                hands.append(
                    HandResult(
                        detected=True,
                        landmarks=lm.landmark,
                        handedness=label,
                        score=score,
                        frame_h=h,
                        frame_w=w,
                    )
                )
        return MultiHandResult(hands=hands, frame_h=h, frame_w=w)

    def draw_landmarks(
        self, frame: np.ndarray, hand: HandResult
    ) -> np.ndarray:
        """Draw hand skeleton on *frame* (in-place)."""
        import mediapipe as mp

        if hand.landmarks is None:
            return frame
        # Reconstruct NormalizedHandLandmarks object
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
