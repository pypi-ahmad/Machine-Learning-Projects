"""Finger Counter Pro — temporal smoothing for stable counts."""

from __future__ import annotations

import collections

from finger_counter import FrameCount


class CountSmoother:
    """Smooth per-hand and total finger counts over time.

    Two layers:
    1. **EMA** (exponential moving average) per hand slot (left / right).
    2. **Majority vote** over a sliding window on the rounded EMA value.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        window: int = 5,
    ) -> None:
        self._alpha = alpha
        self._window = window
        # Per-hand EMA state: "Left" / "Right" → float
        self._ema: dict[str, float] = {}
        # Per-hand vote buffers
        self._votes: dict[str, collections.deque] = {}
        # Total count buffer
        self._total_votes: collections.deque = collections.deque(maxlen=window)
        self._total_ema: float | None = None

    def update(self, frame: FrameCount) -> tuple[dict[str, int], int]:
        """Consume a :class:`FrameCount` and return smoothed results.

        Returns
        -------
        per_hand : dict mapping ``"Left"``/``"Right"`` → smoothed count
        total    : smoothed total across all hands
        """
        per_hand: dict[str, int] = {}

        for state in frame.per_hand:
            key = state.handedness
            raw = state.finger_count

            # EMA
            if key not in self._ema:
                self._ema[key] = float(raw)
            else:
                self._ema[key] = self._alpha * raw + (1 - self._alpha) * self._ema[key]
            rounded = round(self._ema[key])

            # Majority vote
            if key not in self._votes:
                self._votes[key] = collections.deque(maxlen=self._window)
            self._votes[key].append(rounded)
            per_hand[key] = _majority(self._votes[key])

        # Total
        raw_total = frame.total
        if self._total_ema is None:
            self._total_ema = float(raw_total)
        else:
            self._total_ema = self._alpha * raw_total + (1 - self._alpha) * self._total_ema
        rounded_total = round(self._total_ema)
        self._total_votes.append(rounded_total)
        total = _majority(self._total_votes)

        return per_hand, total

    def reset(self) -> None:
        self._ema.clear()
        self._votes.clear()
        self._total_votes.clear()
        self._total_ema = None


def _majority(buf: collections.deque) -> int:
    """Return the most common value in *buf*."""
    if not buf:
        return 0
    counter = collections.Counter(buf)
    return counter.most_common(1)[0][0]
