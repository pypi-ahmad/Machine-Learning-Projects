"""PPE Compliance Monitor — zone-based monitoring.

Maps persons to named polygon zones, overrides per-zone required-PPE
lists, and tracks alert cooldowns.

Usage::

    from zones import ZoneMonitor
    from compliance import Detection

    monitor = ZoneMonitor(cfg)
    zone_results = monitor.process_frame(detections,
                                          compliance_results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from config import PPEConfig, ZoneConfig
from compliance import Detection, PersonCompliance, FrameCompliance


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def point_in_polygon(
    point: tuple[int, int],
    polygon: Sequence[tuple[int, int]],
) -> bool:
    """Ray-casting algorithm — True if *point* is inside *polygon*."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Zone status / alert data classes
# ---------------------------------------------------------------------------

@dataclass
class ZoneStatus:
    """Per-zone aggregated status for one frame."""

    name: str
    polygon: list[tuple[int, int]]
    required_ppe: list[str]
    person_count: int = 0
    compliant_count: int = 0
    violation_count: int = 0
    persons: list[PersonCompliance] = field(default_factory=list)


@dataclass
class AlertEvent:
    """A violation alert to be exported / displayed."""

    zone_name: str
    person_box: tuple[int, int, int, int]
    missing_items: list[str]
    timestamp: float
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Zone monitor
# ---------------------------------------------------------------------------

class ZoneMonitor:
    """Assign persons to zones and produce per-zone compliance stats."""

    def __init__(self, cfg: PPEConfig) -> None:
        self.cfg = cfg
        self._alert_cooldowns: dict[str, float] = {}   # zone_name → last alert ts

    def process_frame(
        self,
        frame_result: FrameCompliance,
    ) -> tuple[list[ZoneStatus], list[AlertEvent]]:
        """Assign each person to zones and build per-zone statuses.

        Returns a list of :class:`ZoneStatus` and new :class:`AlertEvent`
        instances (respecting cooldown).
        """
        now = time.monotonic()
        zones = self.cfg.zones
        default_required = self.cfg.required_ppe

        # If no zones defined → treat whole frame as single virtual zone
        if not zones:
            status = ZoneStatus(
                name="Full-Frame",
                polygon=[],
                required_ppe=default_required,
                person_count=frame_result.total_persons,
                compliant_count=frame_result.compliant_count,
                violation_count=frame_result.violation_count,
                persons=frame_result.persons,
            )
            alerts = self._emit_alerts(status, now)
            return [status], alerts

        statuses: list[ZoneStatus] = []
        all_alerts: list[AlertEvent] = []

        for zone_cfg in zones:
            req = zone_cfg.required_ppe or default_required
            zs = ZoneStatus(
                name=zone_cfg.name,
                polygon=zone_cfg.polygon,
                required_ppe=req,
            )
            for pc in frame_result.persons:
                if point_in_polygon(pc.person.center, zone_cfg.polygon):
                    # Re-evaluate compliance against this zone's requirements
                    missing = [r for r in req if r not in pc.ppe_items]
                    pc_copy = PersonCompliance(
                        person=pc.person,
                        ppe_items=pc.ppe_items,
                        missing_items=missing,
                        is_compliant=len(missing) == 0,
                        zone_name=zone_cfg.name,
                    )
                    zs.persons.append(pc_copy)
                    zs.person_count += 1
                    if pc_copy.is_compliant:
                        zs.compliant_count += 1
                    else:
                        zs.violation_count += 1

            statuses.append(zs)
            all_alerts.extend(self._emit_alerts(zs, now))

        return statuses, all_alerts

    # ---- internal ----------------------------------------------------------

    def _emit_alerts(self, zs: ZoneStatus, now: float) -> list[AlertEvent]:
        """Generate alert events for violations (with cooldown)."""
        alerts: list[AlertEvent] = []
        cooldown = self.cfg.alert_cooldown_sec
        last = self._alert_cooldowns.get(zs.name, 0.0)

        if zs.violation_count == 0:
            return alerts

        if now - last < cooldown:
            return alerts

        for pc in zs.persons:
            if not pc.is_compliant:
                alerts.append(AlertEvent(
                    zone_name=zs.name,
                    person_box=pc.person.box,
                    missing_items=pc.missing_items,
                    timestamp=now,
                    confidence=pc.person.confidence,
                ))

        if alerts:
            self._alert_cooldowns[zs.name] = now

        return alerts
