"""Zone counter — assign detections to zones and check overcrowding.

Core logic module for the Crowd Zone Counter project.

Features
--------
* Point-in-polygon test using ``cv2.pointPolygonTest`` on the
  foot-point of each detected person.
* Per-zone counts.
* Overcrowding alerts when a zone exceeds ``max_capacity``.
* Alert cooldown to suppress repeated warnings.

Usage::

    from zone_counter import ZoneCounter
    from config import load_config

    cfg = load_config("crowd_config.yaml")
    counter = ZoneCounter(cfg)
    result = counter.update(frame_dets)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CrowdConfig, ZoneConfig
from detector import FrameDetections, PersonDetection


@dataclass
class ZoneState:
    """State of a single zone for one frame."""

    name: str
    count: int = 0
    max_capacity: int = 0
    overcrowded: bool = False
    persons_inside: list[PersonDetection] = field(default_factory=list)


@dataclass
class Alert:
    """Overcrowding alert."""

    zone_name: str
    count: int
    max_capacity: int
    frame_idx: int


@dataclass
class FrameResult:
    """Aggregated zone counts and alerts for one frame."""

    zone_states: list[ZoneState] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    total_persons: int = 0
    unzoned_count: int = 0   # persons not inside any zone
    frame_idx: int = 0


class ZoneCounter:
    """Assign person detections to polygon zones and track overcrowding."""

    def __init__(self, cfg: CrowdConfig) -> None:
        self.cfg = cfg
        self._zone_polys: list[tuple[ZoneConfig, np.ndarray]] = [
            (z, np.array(z.polygon, dtype=np.int32)) for z in cfg.zones
        ]
        # Alert cooldown tracking: zone_name → frames remaining
        self._cooldowns: defaultdict[str, int] = defaultdict(int)

    def update(self, dets: FrameDetections) -> FrameResult:
        """Process one frame of person detections.

        Each person's *foot_point* is tested against every zone polygon.
        A person is counted in the **first** zone whose polygon contains
        their foot-point.
        """
        zone_states: list[ZoneState] = []
        alerts: list[Alert] = []
        assigned: set[int] = set()

        for zone_cfg, poly in self._zone_polys:
            state = ZoneState(
                name=zone_cfg.name,
                max_capacity=zone_cfg.max_capacity,
            )

            for i, p in enumerate(dets.persons):
                if i in assigned:
                    continue
                inside = cv2.pointPolygonTest(
                    poly, (float(p.foot_point[0]), float(p.foot_point[1])), False
                )
                if inside >= 0:
                    state.count += 1
                    state.persons_inside.append(p)
                    assigned.add(i)

            # Overcrowding check
            if zone_cfg.max_capacity > 0 and state.count > zone_cfg.max_capacity:
                state.overcrowded = True
                if self._cooldowns[zone_cfg.name] <= 0:
                    alerts.append(Alert(
                        zone_name=zone_cfg.name,
                        count=state.count,
                        max_capacity=zone_cfg.max_capacity,
                        frame_idx=dets.frame_idx,
                    ))
                    self._cooldowns[zone_cfg.name] = self.cfg.alert_cooldown_frames

            zone_states.append(state)

        # Tick cooldowns
        for name in list(self._cooldowns):
            if self._cooldowns[name] > 0:
                self._cooldowns[name] -= 1

        unzoned = dets.total - len(assigned)

        return FrameResult(
            zone_states=zone_states,
            alerts=alerts,
            total_persons=dets.total,
            unzoned_count=unzoned,
            frame_idx=dets.frame_idx,
        )
