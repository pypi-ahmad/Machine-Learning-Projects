"""Head pose estimation for Driver Drowsiness Monitor.

Estimates yaw/pitch/roll from 6 facial landmarks using solvePnP.
Sustained off-center gaze triggers a distraction alert.

Rule explanations:
    - |yaw| > yaw_threshold → looking away left/right
    - |pitch| > pitch_threshold → looking up/down (head nod)
    - Deviation sustained for >= distraction_consec_frames → alert
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DrowsinessConfig
from landmark_detector import (
    CHIN,
    LEFT_EYE_CORNER,
    LEFT_MOUTH_CORNER,
    MODEL_POINTS_3D,
    NOSE_TIP,
    RIGHT_EYE_CORNER,
    RIGHT_MOUTH_CORNER,
    LandmarkResult,
)

log = logging.getLogger("drowsiness.head_pose")


@dataclass
class HeadPoseState:
    """Current head pose state."""

    yaw: float = 0.0           # degrees: negative=left, positive=right
    pitch: float = 0.0         # degrees: negative=down, positive=up
    roll: float = 0.0          # degrees
    off_center: bool = False
    distracted: bool = False
    distraction_frames: int = 0


class HeadPoseEstimator:
    """Estimate head pose and detect distraction."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self._distraction_counter = 0
        self._camera_matrix: np.ndarray | None = None

    def update(self, lm_result: LandmarkResult) -> HeadPoseState:
        """Estimate head pose from landmarks.

        Parameters
        ----------
        lm_result : LandmarkResult
            Current frame landmarks.

        Returns
        -------
        HeadPoseState
        """
        state = HeadPoseState()

        if not lm_result.detected:
            return state

        h, w = lm_result.frame_h, lm_result.frame_w

        # Build camera matrix (approximate)
        if self._camera_matrix is None or self._camera_matrix.shape != (3, 3):
            focal_length = w
            center = (w / 2, h / 2)
            self._camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ], dtype=np.float64)

        # 2D image points (6 landmarks)
        indices = [
            NOSE_TIP, CHIN,
            LEFT_EYE_CORNER, RIGHT_EYE_CORNER,
            LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER,
        ]
        image_points = np.array(
            [lm_result.pixel_coords(i) for i in indices],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            self._camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return state

        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
            cv2.hconcat([pose_mat, np.array([[0, 0, 0, 1]], dtype=np.float64).T]),
        )
        # Fallback: manual Euler extraction from rotation matrix
        yaw, pitch, roll = _rotation_matrix_to_euler(rotation_mat)

        state.yaw = yaw
        state.pitch = pitch
        state.roll = roll

        # Off-center check
        state.off_center = (
            abs(yaw) > self.cfg.yaw_threshold
            or abs(pitch) > self.cfg.pitch_threshold
        )

        # Distraction: sustained off-center
        if state.off_center:
            self._distraction_counter += 1
        else:
            self._distraction_counter = 0

        state.distraction_frames = self._distraction_counter
        state.distracted = (
            self._distraction_counter >= self.cfg.distraction_consec_frames
        )

        return state

    def reset(self) -> None:
        self._distraction_counter = 0


def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to yaw, pitch, roll in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0

    return float(yaw), float(pitch), float(roll)
