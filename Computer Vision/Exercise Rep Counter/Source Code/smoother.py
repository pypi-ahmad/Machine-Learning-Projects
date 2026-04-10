"""Exercise Rep Counter — angle smoothing (EMA)."""

from __future__ import annotations


class AngleSmoother:
    """Exponential moving average on raw joint angles.

    Smoothing is applied *before* stage detection so that
    transient noise doesn't cause false stage transitions.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self._alpha = alpha
        self._prev: float | None = None

    def smooth(self, angle: float) -> float:
        """Return the smoothed angle."""
        if self._prev is None:
            self._prev = angle
        else:
            self._prev = self._alpha * angle + (1 - self._alpha) * self._prev
        return self._prev

    def reset(self) -> None:
        self._prev = None
