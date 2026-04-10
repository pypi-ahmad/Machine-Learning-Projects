"""Possession estimator — nearest-player heuristic.

Determines which player (track ID) "possesses" the ball each frame
based on Euclidean distance from ball centre to each player centre.

The algorithm is deliberately simple and transparent:

1. If a ball is detected, find the nearest player within
   ``possession_radius_px``.
2. If found, that player gains possession.
3. Possession is held for ``possession_hold_frames`` after the ball
   leaves the radius (sticky possession).
4. Cumulative possession time is tracked per player ID.

Usage::

    from possession import PossessionEstimator
    from config import load_config

    cfg = load_config("possession_config.yaml")
    estimator = PossessionEstimator(cfg)
    state = estimator.update(frame_dets)
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PossessionConfig
from tracker import Detection, FrameDetections


@dataclass
class PossessionState:
    """Possession state after processing one frame."""

    current_holder_id: int | None = None
    current_holder_name: str = ""
    ball_detected: bool = False
    ball_centre: tuple[int, int] | None = None
    distance_to_holder: float | None = None
    cumulative_frames: dict[int, int] = field(default_factory=dict)
    frame_idx: int = 0


class PossessionEstimator:
    """Nearest-player possession estimator."""

    def __init__(self, cfg: PossessionConfig) -> None:
        self.cfg = cfg
        self._current_holder_id: int | None = None
        self._hold_countdown: int = 0
        self._cumulative: defaultdict[int, int] = defaultdict(int)
        self._total_frames: int = 0

    def update(self, dets: FrameDetections) -> PossessionState:
        """Process one frame of detections and return possession state."""
        self._total_frames += 1
        state = PossessionState(frame_idx=dets.frame_idx)

        if not dets.balls:
            # No ball — maintain sticky possession
            state.ball_detected = False
            if self._hold_countdown > 0:
                self._hold_countdown -= 1
                state.current_holder_id = self._current_holder_id
            else:
                self._current_holder_id = None
        else:
            state.ball_detected = True
            # Use highest-confidence ball detection
            ball = max(dets.balls, key=lambda b: b.confidence)
            state.ball_centre = ball.centre

            nearest_id, nearest_dist = self._find_nearest_player(
                ball.centre, dets.players
            )

            if nearest_id is not None and nearest_dist <= self.cfg.possession_radius_px:
                self._current_holder_id = nearest_id
                self._hold_countdown = self.cfg.possession_hold_frames
                state.current_holder_id = nearest_id
                state.distance_to_holder = nearest_dist
            elif self._hold_countdown > 0:
                self._hold_countdown -= 1
                state.current_holder_id = self._current_holder_id
            else:
                self._current_holder_id = None

        # Update cumulative counter
        if state.current_holder_id is not None:
            self._cumulative[state.current_holder_id] += 1

        # Build label for the holder
        if state.current_holder_id is not None:
            for p in dets.players:
                if p.track_id == state.current_holder_id:
                    state.current_holder_name = f"Player #{p.track_id}"
                    break

        state.cumulative_frames = dict(self._cumulative)
        return state

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def cumulative(self) -> dict[int, int]:
        return dict(self._cumulative)

    def possession_percentages(self) -> dict[int, float]:
        """Return possession percentage per player ID."""
        if self._total_frames == 0:
            return {}
        return {
            pid: round(frames / self._total_frames * 100, 1)
            for pid, frames in sorted(self._cumulative.items(),
                                       key=lambda x: -x[1])
        }

    def summary(self) -> dict:
        """Return a summary dict for export."""
        pcts = self.possession_percentages()
        contested = self._total_frames - sum(self._cumulative.values())
        return {
            "total_frames": self._total_frames,
            "contested_frames": contested,
            "contested_pct": round(contested / max(self._total_frames, 1) * 100, 1),
            "player_possession": {
                f"player_{pid}": {"frames": self._cumulative[pid], "pct": pcts[pid]}
                for pid in pcts
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_nearest_player(
        ball_centre: tuple[int, int],
        players: list[Detection],
    ) -> tuple[int | None, float]:
        """Return (track_id, distance) of the nearest player, or (None, inf)."""
        bx, by = ball_centre
        best_id: int | None = None
        best_dist = float("inf")

        for p in players:
            if p.track_id < 0:
                continue
            px, py = p.centre
            dist = math.hypot(px - bx, py - by)
            if dist < best_dist:
                best_dist = dist
                best_id = p.track_id

        return best_id, best_dist
