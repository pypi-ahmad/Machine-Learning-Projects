"""Yawn detection via Mouth Aspect Ratio for Driver Drowsiness Monitor.

Computes MAR from inner lip landmarks. A sustained high MAR
indicates a yawn.

MAR formula:
    MAR = mean(vertical_distances) / horizontal_distance
    Vertical: upper-inner lip to lower-inner lip (3 pairs)
    Horizontal: left corner to right corner

Rule explanations:
    - MAR > threshold for >= yawn_consec_frames → yawn event
    - Cooldown prevents duplicate yawn counts within yawn_cooldown_sec
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DrowsinessConfig
from landmark_detector import (
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL_BOT,
    MOUTH_VERTICAL_TOP,
    LandmarkResult,
)

log = logging.getLogger("drowsiness.yawn_tracker")


def compute_mar(lm_result: LandmarkResult) -> float:
    """Compute Mouth Aspect Ratio from inner lip landmarks.

    Parameters
    ----------
    lm_result : LandmarkResult
        Detected landmarks.

    Returns
    -------
    float
        MAR value (typically 0.1–0.3 closed, >0.5 yawning).
    """
    # Vertical distances (3 pairs)
    v_dists = []
    for top_idx, bot_idx in zip(MOUTH_VERTICAL_TOP, MOUTH_VERTICAL_BOT):
        top = np.array(lm_result.pixel_coords(top_idx))
        bot = np.array(lm_result.pixel_coords(bot_idx))
        v_dists.append(np.linalg.norm(top - bot))

    # Horizontal distance
    left = np.array(lm_result.pixel_coords(MOUTH_HORIZONTAL[0]))
    right = np.array(lm_result.pixel_coords(MOUTH_HORIZONTAL[1]))
    h_dist = np.linalg.norm(left - right)

    if h_dist < 1e-6:
        return 0.0
    return float(np.mean(v_dists) / h_dist)


@dataclass
class YawnState:
    """Current yawn tracker state."""

    mar: float = 0.0
    mouth_open: bool = False
    yawn_detected: bool = False
    total_yawns: int = 0
    open_frames: int = 0


class YawnTracker:
    """Track yawns via sustained mouth opening."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self._open_counter = 0
        self._total_yawns = 0
        self._last_yawn_time = 0.0

    def update(self, lm_result: LandmarkResult) -> YawnState:
        """Update yawn state from new landmarks.

        Parameters
        ----------
        lm_result : LandmarkResult
            Current frame landmarks.

        Returns
        -------
        YawnState
        """
        state = YawnState()

        if not lm_result.detected:
            return state

        mar = compute_mar(lm_result)
        state.mar = mar
        state.mouth_open = mar > self.cfg.mar_threshold

        if state.mouth_open:
            self._open_counter += 1
        else:
            self._open_counter = 0

        state.open_frames = self._open_counter

        # Yawn event: sustained opening above threshold
        now = time.monotonic()
        if (
            self._open_counter >= self.cfg.yawn_consec_frames
            and (now - self._last_yawn_time) > self.cfg.yawn_cooldown_sec
        ):
            self._total_yawns += 1
            self._last_yawn_time = now
            state.yawn_detected = True
            log.info("Yawn #%d detected (MAR=%.2f)", self._total_yawns, mar)

        state.total_yawns = self._total_yawns

        return state

    def reset(self) -> None:
        self._open_counter = 0
        self._total_yawns = 0
        self._last_yawn_time = 0.0
