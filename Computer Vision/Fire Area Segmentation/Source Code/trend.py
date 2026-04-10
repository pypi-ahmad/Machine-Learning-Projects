"""Fire Area Segmentation — cross-frame trend summaries.

Maintains a rolling window of per-frame metrics and computes trend
indicators: growth/shrink rate, average coverage, peak values.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from metrics import FrameMetrics


@dataclass
class TrendSummary:
    """Rolling-window trend summary across recent frames."""

    window_size: int
    frames_seen: int

    avg_fire_coverage: float
    avg_smoke_coverage: float
    peak_fire_coverage: float
    peak_smoke_coverage: float
    fire_growth_rate: float     # change per frame (signed)
    smoke_growth_rate: float

    current_fire_coverage: float
    current_smoke_coverage: float


class TrendTracker:
    """Accumulates per-frame metrics and exposes rolling trend summaries."""

    def __init__(self, window: int = 30) -> None:
        self._window = max(window, 2)
        self._history: deque[FrameMetrics] = deque(maxlen=self._window)

    def update(self, m: FrameMetrics) -> TrendSummary:
        """Append a new frame's metrics and return the current trend."""
        self._history.append(m)
        return self._compute()

    def reset(self) -> None:
        self._history.clear()

    # ── internals ──────────────────────────────────────────

    def _compute(self) -> TrendSummary:
        n = len(self._history)
        fire_covs = [m.fire_coverage for m in self._history]
        smoke_covs = [m.smoke_coverage for m in self._history]

        avg_f = sum(fire_covs) / n
        avg_s = sum(smoke_covs) / n
        peak_f = max(fire_covs)
        peak_s = max(smoke_covs)

        # Simple linear growth rate (last vs first in window)
        if n >= 2:
            fire_rate = (fire_covs[-1] - fire_covs[0]) / (n - 1)
            smoke_rate = (smoke_covs[-1] - smoke_covs[0]) / (n - 1)
        else:
            fire_rate = 0.0
            smoke_rate = 0.0

        return TrendSummary(
            window_size=self._window,
            frames_seen=n,
            avg_fire_coverage=round(avg_f, 6),
            avg_smoke_coverage=round(avg_s, 6),
            peak_fire_coverage=round(peak_f, 6),
            peak_smoke_coverage=round(peak_s, 6),
            fire_growth_rate=round(fire_rate, 8),
            smoke_growth_rate=round(smoke_rate, 8),
            current_fire_coverage=fire_covs[-1],
            current_smoke_coverage=smoke_covs[-1],
        )
