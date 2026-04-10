"""Blink counting module for Blink Headpose Analyzer.

Uses the shared EAR utility to count blinks from consecutive
low-EAR frames.  Stateful tracker with reset support.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AnalyzerConfig
from landmark_engine import LandmarkResult
from shared.ear import compute_ear_from_points
from shared.landmarks import LEFT_EYE, RIGHT_EYE, extract_eye_points

log = logging.getLogger("blink_headpose.blink_counter")


@dataclass
class BlinkState:
    """Current blink tracker state for one frame."""

    ear: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    eyes_closed: bool = False
    blink_detected: bool = False
    total_blinks: int = 0
    closure_frames: int = 0


class BlinkCounter:
    """Count blinks from EAR-based eye closure tracking."""

    def __init__(self, cfg: AnalyzerConfig) -> None:
        self.cfg = cfg
        self._closure_counter = 0
        self._total_blinks = 0

    def update(self, lm: LandmarkResult) -> BlinkState:
        """Update blink state from new landmarks.

        Parameters
        ----------
        lm : LandmarkResult
            Current frame landmarks.

        Returns
        -------
        BlinkState
        """
        state = BlinkState()
        if not lm.detected:
            return state

        # Extract eye points via shared utility
        left_pts = extract_eye_points(
            lm.landmarks, LEFT_EYE, lm.frame_w, lm.frame_h,
        )
        right_pts = extract_eye_points(
            lm.landmarks, RIGHT_EYE, lm.frame_w, lm.frame_h,
        )

        # Compute EAR via shared utility
        left_ear = compute_ear_from_points(left_pts)
        right_ear = compute_ear_from_points(right_pts)
        ear = (left_ear + right_ear) / 2.0

        state.ear = ear
        state.left_ear = left_ear
        state.right_ear = right_ear
        state.eyes_closed = ear < self.cfg.ear_threshold

        # Blink detection: count consecutive closed frames
        if state.eyes_closed:
            self._closure_counter += 1
        else:
            if self._closure_counter >= self.cfg.blink_consec_frames:
                self._total_blinks += 1
                state.blink_detected = True
            self._closure_counter = 0

        state.closure_frames = self._closure_counter
        state.total_blinks = self._total_blinks
        return state

    def reset(self) -> None:
        """Reset all tracking state."""
        self._closure_counter = 0
        self._total_blinks = 0
