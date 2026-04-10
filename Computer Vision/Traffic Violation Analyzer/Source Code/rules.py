"""Traffic Violation Analyzer — rule engine.

Evaluates line crossings and wrong-way movement using per-track position
histories maintained by :class:`tracker.TrackManager`.

Supported rules:
1. **Line crossing** — counts each time a tracked vehicle's center
   crosses a virtual line.
2. **Wrong-way detection** — flags a crossing whose direction is opposite
   to the line's declared *allowed* direction.

Usage::

    from rules import RuleEngine, ViolationEvent

    engine = RuleEngine(cfg)
    events = engine.evaluate(detections, track_manager)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from config import TrafficConfig, LineConfig, ZoneConfig
from detector import Detection
from tracker import TrackManager


# ---------------------------------------------------------------------------
# Event data classes
# ---------------------------------------------------------------------------

@dataclass
class ViolationEvent:
    """A violation or counting event emitted by the rule engine."""

    event_type: str          # "line_cross" | "wrong_way" | "zone_entry"
    line_or_zone: str        # name of the line / zone
    track_id: int | None
    class_name: str
    direction: str           # detected movement direction
    timestamp: float         # monotonic seconds
    frame_idx: int = 0
    confidence: float = 0.0
    center: tuple[int, int] = (0, 0)


@dataclass
class FrameEvents:
    """Aggregate events for one frame."""

    events: list[ViolationEvent] = field(default_factory=list)
    line_counts: dict[str, int] = field(default_factory=dict)
    wrong_way_count: int = 0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cross_product_sign(
    a: tuple[int, int],
    b: tuple[int, int],
    p: tuple[int, int],
) -> float:
    """Sign of the cross-product (b-a) × (p-a).

    Positive → p is to the left of a→b.
    Negative → p is to the right.
    Zero     → collinear.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _infer_direction(prev: tuple[int, int], curr: tuple[int, int]) -> str:
    """Dominant cardinal direction from *prev* to *curr*."""
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def point_in_polygon(
    point: tuple[int, int],
    polygon: Sequence[tuple[int, int]],
) -> bool:
    """Ray-casting point-in-polygon test."""
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
# Rule engine
# ---------------------------------------------------------------------------

class RuleEngine:
    """Evaluate traffic rules on tracked detections."""

    def __init__(self, cfg: TrafficConfig) -> None:
        self.cfg = cfg
        # track_id → set of line names already crossed (prevents double-count)
        self._crossed: dict[int, set[str]] = {}
        # total crossing counts per line
        self._line_counts: dict[str, int] = {ln.name: 0 for ln in cfg.lines}

    @property
    def line_counts(self) -> dict[str, int]:
        return dict(self._line_counts)

    # ---- public API --------------------------------------------------------

    def evaluate(
        self,
        detections: list[Detection],
        tm: TrackManager,
        frame_idx: int = 0,
    ) -> FrameEvents:
        """Check all rules and return events for this frame."""
        now = time.monotonic()
        events: list[ViolationEvent] = []
        wrong_way = 0

        vehicle_classes = set(self.cfg.vehicle_classes)

        for det in detections:
            if det.track_id is None:
                continue
            if det.class_name not in vehicle_classes:
                continue

            trail = tm.get_trail(det.track_id)
            if len(trail) < 2:
                continue

            prev = trail[-2]
            curr = trail[-1]

            # Ensure per-track state exists
            if det.track_id not in self._crossed:
                self._crossed[det.track_id] = set()

            # ── Line-crossing check ────────────────────────
            for line_cfg in self.cfg.lines:
                if line_cfg.name in self._crossed[det.track_id]:
                    continue  # already counted

                side_prev = _cross_product_sign(line_cfg.pt1, line_cfg.pt2, prev)
                side_curr = _cross_product_sign(line_cfg.pt1, line_cfg.pt2, curr)

                if side_prev * side_curr < 0:
                    # Crossed!
                    self._crossed[det.track_id].add(line_cfg.name)
                    direction = _infer_direction(prev, curr)
                    self._line_counts[line_cfg.name] = self._line_counts.get(line_cfg.name, 0) + 1

                    evt = ViolationEvent(
                        event_type="line_cross",
                        line_or_zone=line_cfg.name,
                        track_id=det.track_id,
                        class_name=det.class_name,
                        direction=direction,
                        timestamp=now,
                        frame_idx=frame_idx,
                        confidence=det.confidence,
                        center=det.center,
                    )
                    events.append(evt)

                    # Wrong-way check
                    if line_cfg.direction != "any" and direction != line_cfg.direction:
                        ww = ViolationEvent(
                            event_type="wrong_way",
                            line_or_zone=line_cfg.name,
                            track_id=det.track_id,
                            class_name=det.class_name,
                            direction=direction,
                            timestamp=now,
                            frame_idx=frame_idx,
                            confidence=det.confidence,
                            center=det.center,
                        )
                        events.append(ww)
                        wrong_way += 1

        return FrameEvents(
            events=events,
            line_counts=dict(self._line_counts),
            wrong_way_count=wrong_way,
        )
