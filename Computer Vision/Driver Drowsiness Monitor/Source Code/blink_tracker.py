"""Blink and eye-closure tracking for Driver Drowsiness Monitor.

Computes Eye Aspect Ratio (EAR) from 6-point eye contours,
detects individual blinks, prolonged eye closure, and PERCLOS
(percentage of time eyes are closed over a rolling window).

EAR formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    where p1..p6 are the 6 eye contour landmarks.

Rule explanations:
    - EAR < threshold for >= blink_consec_frames → one blink counted
    - EAR < threshold for >= drowsy_eye_frames → prolonged closure alert
    - PERCLOS > perclos_threshold over window → fatigue alert
"""

from __future__ import annotations

import collections
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DrowsinessConfig
from landmark_detector import LEFT_EYE, RIGHT_EYE, LandmarkResult

log = logging.getLogger("drowsiness.blink_tracker")


def compute_ear(lm_result: LandmarkResult, eye_indices: list[int]) -> float:
    """Compute Eye Aspect Ratio from landmark result.

    Parameters
    ----------
    lm_result : LandmarkResult
        Detected landmarks.
    eye_indices : list[int]
        6 MediaPipe landmark indices for one eye.

    Returns
    -------
    float
        EAR value (typically 0.15–0.35 open, <0.21 closed).
    """
    pts = [lm_result.pixel_coords(i) for i in eye_indices]

    # Vertical distances
    v1 = np.linalg.norm(
        np.array(pts[1]) - np.array(pts[5]),
    )
    v2 = np.linalg.norm(
        np.array(pts[2]) - np.array(pts[4]),
    )
    # Horizontal distance
    h = np.linalg.norm(
        np.array(pts[0]) - np.array(pts[3]),
    )
    if h < 1e-6:
        return 0.3  # fallback to "open"
    return float((v1 + v2) / (2.0 * h))


@dataclass
class BlinkState:
    """Current blink tracker state."""

    ear: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    eyes_closed: bool = False
    blink_detected: bool = False
    total_blinks: int = 0
    closure_frames: int = 0
    prolonged_closure: bool = False
    perclos: float = 0.0
    drowsy_by_perclos: bool = False


class BlinkTracker:
    """Track blinks, prolonged closure, and PERCLOS."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self._closure_counter = 0
        self._total_blinks = 0
        # Rolling window for PERCLOS (stores 1=closed, 0=open per frame)
        self._perclos_window: collections.deque[tuple[float, bool]] = (
            collections.deque()
        )

    def update(self, lm_result: LandmarkResult) -> BlinkState:
        """Update blink state from new landmarks.

        Parameters
        ----------
        lm_result : LandmarkResult
            Current frame landmarks.

        Returns
        -------
        BlinkState
        """
        state = BlinkState()

        if not lm_result.detected:
            return state

        # Compute EAR for each eye, average
        left_ear = compute_ear(lm_result, LEFT_EYE)
        right_ear = compute_ear(lm_result, RIGHT_EYE)
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

        # Prolonged closure → drowsiness
        state.prolonged_closure = (
            self._closure_counter >= self.cfg.drowsy_eye_frames
        )

        # PERCLOS: fraction of time eyes closed in rolling window
        now = time.monotonic()
        self._perclos_window.append((now, state.eyes_closed))

        # Trim old entries
        cutoff = now - self.cfg.perclos_window_sec
        while self._perclos_window and self._perclos_window[0][0] < cutoff:
            self._perclos_window.popleft()

        if self._perclos_window:
            closed_count = sum(1 for _, c in self._perclos_window if c)
            state.perclos = closed_count / len(self._perclos_window)
        state.drowsy_by_perclos = state.perclos > self.cfg.perclos_threshold

        return state

    def reset(self) -> None:
        """Reset all tracking state."""
        self._closure_counter = 0
        self._total_blinks = 0
        self._perclos_window.clear()
