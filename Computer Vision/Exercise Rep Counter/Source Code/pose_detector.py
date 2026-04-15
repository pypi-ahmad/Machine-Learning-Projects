"""Exercise Rep Counter -- MediaPipe Pose wrapper (33 body landmarks)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# COCO-compatible landmark indices used for exercises
# (MediaPipe Pose provides 33 landmarks)
class LM:
    """MediaPipe Pose landmark indices."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


NUM_LANDMARKS = 33


@dataclasses.dataclass
class PoseResult:
    """Detection result for a single body pose."""

    detected: bool
    landmarks: list | None   # NormalizedLandmark list (len 33)
    visibility: list[float]  # per-landmark visibility scores
    frame_h: int
    frame_w: int

    def pixel(self, idx: int) -> tuple[int, int]:
        """Return (x, y) pixel coordinates for landmark *idx*."""
        lm = self.landmarks[idx]
        return int(lm.x * self.frame_w), int(lm.y * self.frame_h)

    def norm(self, idx: int) -> tuple[float, float, float]:
        """Return (x, y, z) normalised coordinates."""
        lm = self.landmarks[idx]
        return lm.x, lm.y, lm.z

    def vis(self, idx: int) -> float:
        """Return visibility score for landmark *idx*."""
        return self.visibility[idx]


class PoseDetector:
    """Lazy-loading MediaPipe Pose wrapper."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._complexity = model_complexity
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._static = static_image_mode
        self._pose = None
        self._mp_pose = None
        self._mp_draw = None

    def load(self) -> None:
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._pose = self._mp_pose.Pose(
            static_image_mode=self._static,
            model_complexity=self._complexity,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )

    @property
    def ready(self) -> bool:
        return self._pose is not None

    def detect(self, frame: np.ndarray) -> PoseResult:
        """Detect body pose and return :class:`PoseResult`."""
        import cv2

        if not self.ready:
            self.load()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            vis = [lm.visibility for lm in lms]
            return PoseResult(
                detected=True,
                landmarks=lms,
                visibility=vis,
                frame_h=h,
                frame_w=w,
            )
        return PoseResult(
            detected=False,
            landmarks=None,
            visibility=[],
            frame_h=h,
            frame_w=w,
        )

    def draw_skeleton(self, frame: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Draw pose skeleton on *frame* (in-place)."""
        import mediapipe as mp

        if pose.landmarks is None:
            return frame
        lm_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for pt in pose.landmarks:
            lm = lm_proto.landmark.add()
            lm.x, lm.y, lm.z = pt.x, pt.y, pt.z
            lm.visibility = pt.visibility
        self._mp_draw.draw_landmarks(
            frame,
            lm_proto,
            self._mp_pose.POSE_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self._mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
        )
        return frame

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
