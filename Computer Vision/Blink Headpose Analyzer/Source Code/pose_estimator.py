"""Head pose estimation module for Blink Headpose Analyzer.

Uses the shared head-pose math utility to estimate yaw/pitch/roll
from 6 facial landmarks via solvePnP.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AnalyzerConfig
from landmark_engine import LandmarkResult
from shared.head_pose_math import solve_head_pose
from shared.landmarks import extract_pose_points

log = logging.getLogger("blink_headpose.pose_estimator")


@dataclass
class PoseState:
    """Current head pose state for one frame."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    off_center: bool = False


class PoseEstimator:
    """Estimate head orientation from facial landmarks."""

    def __init__(self, cfg: AnalyzerConfig) -> None:
        self.cfg = cfg

    def update(self, lm: LandmarkResult) -> PoseState:
        """Estimate head pose from landmarks.

        Parameters
        ----------
        lm : LandmarkResult
            Current frame landmarks.

        Returns
        -------
        PoseState
        """
        state = PoseState()
        if not lm.detected:
            return state

        # Extract 6 head-pose landmark points via shared utility
        image_points = extract_pose_points(
            lm.landmarks, lm.frame_w, lm.frame_h,
        )

        result = solve_head_pose(image_points, lm.frame_w, lm.frame_h)
        if result is None:
            return state

        yaw, pitch, roll = result
        state.yaw = yaw
        state.pitch = pitch
        state.roll = roll
        state.off_center = (
            abs(yaw) > self.cfg.yaw_threshold
            or abs(pitch) > self.cfg.pitch_threshold
        )
        return state
