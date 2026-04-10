"""Zone counting and low-stock detection logic.

Pure logic module — no I/O, no OpenCV rendering.  Operates on lists
of detections and zone definitions, producing structured counts and
alert records.

Usage::

    from zones import ZoneCounter

    counter = ZoneCounter(zones=cfg.zones)
    result  = counter.update(detections)
    # result.zone_counts, result.alerts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from config import ZoneConfig


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------
def point_in_polygon(px: int, py: int, polygon: Sequence[tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test.

    Parameters
    ----------
    px, py : int
        Point coordinates.
    polygon : sequence of (x, y) tuples
        Ordered vertices of the polygon (closed automatically).
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single object detection."""

    box: tuple[int, int, int, int]   # x1, y1, x2, y2
    center: tuple[int, int]          # cx, cy
    class_name: str
    confidence: float
    class_id: int = 0


@dataclass
class ZoneStatus:
    """Counting result for one zone."""

    name: str
    count: int
    threshold: int
    is_low_stock: bool
    class_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class FrameResult:
    """Full result for a single frame."""

    detections: list[Detection]
    total_count: int
    zone_statuses: list[ZoneStatus]
    alerts: list[str]
    timestamp: str = ""


@dataclass
class AlertEvent:
    """A logged low-stock event."""

    timestamp: str
    zone: str
    count: int
    threshold: int
    event_type: str = "low_stock"


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------
class ZoneCounter:
    """Counts detections per zone and generates low-stock alerts.

    Parameters
    ----------
    zones : list[ZoneConfig]
        Zone definitions loaded from configuration.
    default_threshold : int
        Fallback threshold when a zone does not specify one.
    """

    def __init__(self, zones: list[ZoneConfig], default_threshold: int = 3) -> None:
        self._zones = zones
        self._default_threshold = default_threshold
        self.events: list[AlertEvent] = []

    @property
    def zone_configs(self) -> list[ZoneConfig]:
        return self._zones

    def update(self, detections: list[Detection]) -> FrameResult:
        """Process detections against configured zones.

        Returns a :class:`FrameResult` with per-zone counts and alerts.
        """
        now = datetime.now().isoformat(timespec="seconds")
        zone_statuses: list[ZoneStatus] = []
        alerts: list[str] = []

        for zone in self._zones:
            threshold = zone.low_stock_threshold or self._default_threshold
            class_counts: dict[str, int] = {}
            count = 0

            for det in detections:
                # Class filter
                if zone.classes and det.class_name not in zone.classes:
                    continue
                if point_in_polygon(det.center[0], det.center[1], zone.polygon):
                    count += 1
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

            is_low = count < threshold
            zone_statuses.append(ZoneStatus(
                name=zone.name,
                count=count,
                threshold=threshold,
                is_low_stock=is_low,
                class_counts=class_counts,
            ))

            if is_low:
                msg = f"LOW STOCK: {zone.name} ({count}/{threshold} items)"
                alerts.append(msg)
                self.events.append(AlertEvent(
                    timestamp=now,
                    zone=zone.name,
                    count=count,
                    threshold=threshold,
                ))

        return FrameResult(
            detections=detections,
            total_count=len(detections),
            zone_statuses=zone_statuses,
            alerts=alerts,
            timestamp=now,
        )

    def clear_events(self) -> None:
        """Reset accumulated event log."""
        self.events.clear()
