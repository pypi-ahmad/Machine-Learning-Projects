"""Wound Area Measurement — multi-image change tracking.

When a sequence of images of the same wound is provided (e.g. healing
over days), this module tracks relative area changes across the series.

**Disclaimer**: All measurements are *relative pixel-based estimates* and
must NOT be used for clinical diagnosis.  See README for full disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from metrics import WoundMetrics


@dataclass
class ChangeEntry:
    """One timestep in the wound tracking series."""

    index: int
    source: str
    wound_area_px: int
    wound_coverage: float
    delta_px: int               # change vs previous  (+ve = grew)
    delta_ratio: float          # change / previous area  (signed)


@dataclass
class ChangeSummary:
    """Summary across the entire series."""

    entries: list[ChangeEntry] = field(default_factory=list)
    total_images: int = 0
    initial_area_px: int = 0
    final_area_px: int = 0
    net_change_px: int = 0
    net_change_ratio: float = 0.0   # (final - initial) / initial  (signed)
    peak_area_px: int = 0


class ChangeTracker:
    """Accumulates per-image metrics and computes a change summary."""

    def __init__(self) -> None:
        self._entries: list[ChangeEntry] = []
        self._prev_area: int | None = None

    def add(self, m: WoundMetrics, source: str = "") -> ChangeEntry:
        """Record one image's metrics and return the change entry."""
        idx = len(self._entries)
        delta = 0
        delta_ratio = 0.0
        if self._prev_area is not None and self._prev_area > 0:
            delta = m.wound_area_px - self._prev_area
            delta_ratio = round(delta / self._prev_area, 6)

        entry = ChangeEntry(
            index=idx,
            source=source,
            wound_area_px=m.wound_area_px,
            wound_coverage=m.wound_coverage,
            delta_px=delta,
            delta_ratio=delta_ratio,
        )
        self._entries.append(entry)
        self._prev_area = m.wound_area_px
        return entry

    def summary(self) -> ChangeSummary:
        """Return a summary of the tracked series."""
        if not self._entries:
            return ChangeSummary()

        initial = self._entries[0].wound_area_px
        final = self._entries[-1].wound_area_px
        net = final - initial
        net_ratio = round(net / initial, 6) if initial > 0 else 0.0
        peak = max(e.wound_area_px for e in self._entries)

        return ChangeSummary(
            entries=list(self._entries),
            total_images=len(self._entries),
            initial_area_px=initial,
            final_area_px=final,
            net_change_px=net,
            net_change_ratio=net_ratio,
            peak_area_px=peak,
        )

    def reset(self) -> None:
        self._entries.clear()
        self._prev_area = None
