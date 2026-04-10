"""Duplicate suppression tracker for plate reads across frames.

Prevents the same plate from being logged repeatedly within a
configurable cooldown window.

Usage::

    from tracker import PlateTracker
    from config import PlateConfig

    tracker = PlateTracker(PlateConfig())
    is_new = tracker.is_new("AB 123 CD")
"""

from __future__ import annotations

import time
from collections import OrderedDict

from config import PlateConfig


class PlateTracker:
    """Suppress duplicate plate reads within a cooldown window."""

    def __init__(self, cfg: PlateConfig) -> None:
        self.cfg = cfg
        self._seen: OrderedDict[str, float] = OrderedDict()

    def is_new(self, plate: str) -> bool:
        """Return True if *plate* has not been seen within the cooldown."""
        if not self.cfg.dedup_enabled or not plate:
            return True

        now = time.time()
        self._evict_stale(now)

        if plate in self._seen:
            if now - self._seen[plate] < self.cfg.dedup_cooldown:
                return False

        self._seen[plate] = now

        # Cap memory
        while len(self._seen) > self.cfg.dedup_max_entries:
            self._seen.popitem(last=False)

        return True

    def reset(self) -> None:
        """Clear all tracked plates."""
        self._seen.clear()

    @property
    def tracked_count(self) -> int:
        return len(self._seen)

    def _evict_stale(self, now: float) -> None:
        """Remove entries older than 10× cooldown."""
        max_age = self.cfg.dedup_cooldown * 10
        while self._seen:
            oldest_plate = next(iter(self._seen))
            if now - self._seen[oldest_plate] > max_age:
                self._seen.popitem(last=False)
            else:
                break
