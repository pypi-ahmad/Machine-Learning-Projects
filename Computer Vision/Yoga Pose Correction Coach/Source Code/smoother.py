"""Yoga Pose Correction Coach — temporal smoothing for pose label."""

from __future__ import annotations

import collections


class PoseSmoother:
    """Majority-vote smoothing over a sliding window of pose labels."""

    def __init__(self, window: int = 7) -> None:
        self._window = window
        self._buf: collections.deque = collections.deque(maxlen=window)

    def update(self, label: str) -> str:
        """Add *label* and return the smoothed (majority-vote) label."""
        self._buf.append(label)
        counter = collections.Counter(self._buf)
        return counter.most_common(1)[0][0]

    def reset(self) -> None:
        self._buf.clear()
