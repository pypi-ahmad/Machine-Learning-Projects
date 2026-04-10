"""Video Event Search — event generator.

Takes tracked detections + zone/line configuration and emits structured
:class:`Event` objects:

* **APPEAR** — track ID first seen
* **DISAPPEAR** — track ID gone for ``disappear_frames``
* **ZONE_ENTER / ZONE_EXIT** — centroid enters/leaves a polygon zone
* **LINE_CROSS** — centroid crosses a virtual line
* **DWELL** — object stays inside a zone longer than ``dwell_threshold``

Usage::

    from event_generator import EventGenerator

    gen = EventGenerator(cfg, fps=25.0)
    events = gen.process(detections, track_manager, frame_idx)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import cv2
import numpy as np

from config import (
    Event,
    EventSearchConfig,
    EventType,
    LineConfig,
    ZoneConfig,
)
from detector import Detection
from tracker import TrackManager


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cross_product_sign(
    a: tuple[int, int],
    b: tuple[int, int],
    p: tuple[int, int],
) -> float:
    """Sign of (b-a) × (p-a)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _infer_direction(prev: tuple[int, int], curr: tuple[int, int]) -> str:
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def _point_in_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    """Ray-casting point-in-polygon test."""
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


# ---------------------------------------------------------------------------
# Event generator
# ---------------------------------------------------------------------------

class EventGenerator:
    """Evaluate detection trajectories and emit structured events."""

    def __init__(self, cfg: EventSearchConfig, fps: float = 25.0) -> None:
        self.cfg = cfg
        self.fps = max(fps, 1.0)

        # Per-track state
        self._appeared: set[int] = set()         # tracks that got APPEAR
        self._disappeared: set[int] = set()       # tracks that got DISAPPEAR
        self._last_seen_frame: dict[int, int] = {}
        self._last_class: dict[int, str] = {}
        self._last_conf: dict[int, float] = {}
        self._last_center: dict[int, tuple[int, int]] = {}

        # Zone state: track_id → zone_name → bool (inside)
        self._in_zone: dict[int, dict[str, bool]] = defaultdict(dict)
        # Zone dwell: track_id → zone_name → first_frame_inside
        self._zone_enter_frame: dict[int, dict[str, int]] = defaultdict(dict)
        # Dwell events already emitted: (track_id, zone_name)
        self._dwell_emitted: set[tuple[int, str]] = set()

        # Line crossing: track_id → set of line names already crossed
        self._crossed: dict[int, set[str]] = defaultdict(set)

    def process(
        self,
        detections: list[Detection],
        tm: TrackManager,
        frame_idx: int,
    ) -> list[Event]:
        """Generate events for one frame."""
        events: list[Event] = []
        ts = frame_idx / self.fps

        target_classes = set(self.cfg.target_classes) if self.cfg.target_classes else None

        # Index current detections by track_id
        det_by_id: dict[int, Detection] = {}
        for det in detections:
            if det.track_id is None:
                continue
            if target_classes and det.class_name not in target_classes:
                continue
            det_by_id[det.track_id] = det
            self._last_seen_frame[det.track_id] = frame_idx
            self._last_class[det.track_id] = det.class_name
            self._last_conf[det.track_id] = det.confidence
            self._last_center[det.track_id] = det.center

        # --- APPEAR events ---
        for tid in tm.newly_appeared:
            if tid in self._disappeared:
                self._disappeared.discard(tid)
            if tid not in self._appeared:
                self._appeared.add(tid)
                det = det_by_id.get(tid)
                if det:
                    events.append(Event(
                        event_type=EventType.APPEAR.value,
                        track_id=tid,
                        class_name=det.class_name,
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        confidence=det.confidence,
                        center=det.center,
                    ))

        # --- DISAPPEAR events ---
        for tid in tm.newly_disappeared:
            if tid in self._appeared and tid not in self._disappeared:
                last_frame = self._last_seen_frame.get(tid, frame_idx)
                if frame_idx - last_frame >= 0:
                    self._disappeared.add(tid)
                    events.append(Event(
                        event_type=EventType.DISAPPEAR.value,
                        track_id=tid,
                        class_name=self._last_class.get(tid, "unknown"),
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        confidence=self._last_conf.get(tid, 0.0),
                        center=self._last_center.get(tid, (0, 0)),
                    ))

        # --- Zone and line events for active tracks ---
        for tid, det in det_by_id.items():
            trail = tm.get_trail(tid)

            # Zone enter / exit / dwell
            for zone_cfg in self.cfg.zones:
                inside = _point_in_polygon(det.center, zone_cfg.polygon)
                was_inside = self._in_zone[tid].get(zone_cfg.name, False)

                if inside and not was_inside:
                    # ZONE_ENTER
                    self._in_zone[tid][zone_cfg.name] = True
                    self._zone_enter_frame[tid][zone_cfg.name] = frame_idx
                    self._dwell_emitted.discard((tid, zone_cfg.name))
                    events.append(Event(
                        event_type=EventType.ZONE_ENTER.value,
                        track_id=tid,
                        class_name=det.class_name,
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        confidence=det.confidence,
                        center=det.center,
                        zone_or_line=zone_cfg.name,
                    ))

                elif not inside and was_inside:
                    # ZONE_EXIT
                    self._in_zone[tid][zone_cfg.name] = False
                    enter_frame = self._zone_enter_frame[tid].pop(zone_cfg.name, frame_idx)
                    dwell_sec = (frame_idx - enter_frame) / self.fps
                    events.append(Event(
                        event_type=EventType.ZONE_EXIT.value,
                        track_id=tid,
                        class_name=det.class_name,
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        confidence=det.confidence,
                        center=det.center,
                        zone_or_line=zone_cfg.name,
                        dwell_seconds=dwell_sec,
                    ))

                elif inside and was_inside:
                    # DWELL check
                    key = (tid, zone_cfg.name)
                    if key not in self._dwell_emitted:
                        enter_frame = self._zone_enter_frame[tid].get(zone_cfg.name, frame_idx)
                        dwell_sec = (frame_idx - enter_frame) / self.fps
                        if dwell_sec >= self.cfg.dwell_threshold:
                            self._dwell_emitted.add(key)
                            events.append(Event(
                                event_type=EventType.DWELL.value,
                                track_id=tid,
                                class_name=det.class_name,
                                frame_idx=frame_idx,
                                timestamp_sec=ts,
                                confidence=det.confidence,
                                center=det.center,
                                zone_or_line=zone_cfg.name,
                                dwell_seconds=dwell_sec,
                            ))

            # Line crossing
            if len(trail) >= 2:
                prev = trail[-2]
                curr = trail[-1]
                for line_cfg in self.cfg.lines:
                    if line_cfg.name in self._crossed[tid]:
                        continue
                    side_prev = _cross_product_sign(line_cfg.pt1, line_cfg.pt2, prev)
                    side_curr = _cross_product_sign(line_cfg.pt1, line_cfg.pt2, curr)
                    if side_prev * side_curr < 0:
                        self._crossed[tid].add(line_cfg.name)
                        direction = _infer_direction(prev, curr)
                        events.append(Event(
                            event_type=EventType.LINE_CROSS.value,
                            track_id=tid,
                            class_name=det.class_name,
                            frame_idx=frame_idx,
                            timestamp_sec=ts,
                            confidence=det.confidence,
                            center=det.center,
                            zone_or_line=line_cfg.name,
                            direction=direction,
                        ))

        return events
