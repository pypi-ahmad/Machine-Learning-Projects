"""Temporal smoothing for Gaze Direction Estimator.

Reduces jitter in gaze classification via two mechanisms:

1. **Exponential Moving Average (EMA)** on iris ratios.
2. **Majority vote** over a sliding window of classified
   directions.
"""

from __future__ import annotations

import collections
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GazeConfig
from gaze_classifier import CENTER


@dataclass
class SmoothedState:
    """Smoothed gaze output."""

    raw_direction: str = CENTER
    smoothed_direction: str = CENTER
    raw_h: float = 0.5
    raw_v: float = 0.5
    smoothed_h: float = 0.5
    smoothed_v: float = 0.5


class GazeSmoother:
    """EMA + majority-vote smoother for gaze ratios and direction."""

    def __init__(self, cfg: GazeConfig) -> None:
        self.cfg = cfg
        self._ema_h: float | None = None
        self._ema_v: float | None = None
        self._vote_window: collections.deque[str] = collections.deque(
            maxlen=cfg.vote_window,
        )

    def update(
        self,
        h_ratio: float,
        v_ratio: float,
        direction: str,
    ) -> SmoothedState:
        """Apply smoothing to new gaze data.

        Parameters
        ----------
        h_ratio, v_ratio : float
            Raw iris position ratios.
        direction : str
            Raw classified direction.

        Returns
        -------
        SmoothedState
        """
        state = SmoothedState(
            raw_direction=direction,
            raw_h=h_ratio,
            raw_v=v_ratio,
        )

        if not self.cfg.enable_smoothing:
            state.smoothed_direction = direction
            state.smoothed_h = h_ratio
            state.smoothed_v = v_ratio
            return state

        # EMA on ratios
        alpha = self.cfg.ema_alpha
        if self._ema_h is None:
            self._ema_h = h_ratio
            self._ema_v = v_ratio
        else:
            self._ema_h = alpha * h_ratio + (1 - alpha) * self._ema_h
            self._ema_v = alpha * v_ratio + (1 - alpha) * self._ema_v

        state.smoothed_h = self._ema_h
        state.smoothed_v = self._ema_v

        # Majority vote on direction
        self._vote_window.append(direction)
        vote_counts: dict[str, int] = {}
        for d in self._vote_window:
            vote_counts[d] = vote_counts.get(d, 0) + 1

        state.smoothed_direction = max(vote_counts, key=vote_counts.get)

        return state

    def reset(self) -> None:
        """Clear smoothing state."""
        self._ema_h = None
        self._ema_v = None
        self._vote_window.clear()
