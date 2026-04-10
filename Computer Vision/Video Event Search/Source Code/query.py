"""Video Event Search — query / search interface.

Filter and search stored events by type, time range, track ID, zone,
class, and free-text keyword.

Usage::

    from query import EventQuery

    q = EventQuery("outputs/events.json")
    results = q.search(event_type="zone_enter", class_name="person")
    results = q.search(time_range=(5.0, 20.0), zone="crosswalk_zone")
    summary = q.summary()
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class EventQuery:
    """Filter / search interface over a JSON event store."""

    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        self._events: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.json_path.exists():
            self._events = json.loads(
                self.json_path.read_text(encoding="utf-8"),
            )

    def reload(self) -> None:
        """Re-read events from disk."""
        self._load()

    @property
    def total(self) -> int:
        return len(self._events)

    def search(
        self,
        *,
        event_type: str | list[str] | None = None,
        track_id: int | list[int] | None = None,
        class_name: str | list[str] | None = None,
        zone: str | list[str] | None = None,
        direction: str | None = None,
        time_range: tuple[float, float] | None = None,
        frame_range: tuple[int, int] | None = None,
        min_confidence: float = 0.0,
        min_dwell: float = 0.0,
        keyword: str | None = None,
        limit: int = 0,
    ) -> list[dict]:
        """Return events matching all supplied filters (AND logic)."""
        results: list[dict] = []

        event_types = _normalise_list(event_type)
        track_ids = _normalise_int_list(track_id)
        class_names = _normalise_list(class_name)
        zones = _normalise_list(zone)

        for evt in self._events:
            if event_types and evt.get("event_type") not in event_types:
                continue
            if track_ids and evt.get("track_id") not in track_ids:
                continue
            if class_names and evt.get("class_name") not in class_names:
                continue
            if zones and evt.get("zone_or_line") not in zones:
                continue
            if direction and evt.get("direction") != direction:
                continue
            if time_range:
                ts = evt.get("timestamp_sec", 0.0)
                if ts < time_range[0] or ts > time_range[1]:
                    continue
            if frame_range:
                fi = evt.get("frame_idx", 0)
                if fi < frame_range[0] or fi > frame_range[1]:
                    continue
            if min_confidence and evt.get("confidence", 0.0) < min_confidence:
                continue
            if min_dwell and evt.get("dwell_seconds", 0.0) < min_dwell:
                continue
            if keyword:
                blob = json.dumps(evt, default=str).lower()
                if keyword.lower() not in blob:
                    continue

            results.append(evt)
            if limit and len(results) >= limit:
                break

        return results

    def summary(self) -> dict[str, Any]:
        """Return an overview of the event store contents."""
        type_counts = Counter(e.get("event_type", "?") for e in self._events)
        class_counts = Counter(e.get("class_name", "?") for e in self._events)
        zone_counts = Counter(
            e.get("zone_or_line", "")
            for e in self._events
            if e.get("zone_or_line")
        )
        track_ids = sorted({e.get("track_id", -1) for e in self._events})

        timestamps = [e.get("timestamp_sec", 0.0) for e in self._events]
        t_min = min(timestamps) if timestamps else 0.0
        t_max = max(timestamps) if timestamps else 0.0

        return {
            "total_events": len(self._events),
            "time_span_sec": round(t_max - t_min, 2),
            "event_types": dict(type_counts),
            "class_counts": dict(class_counts),
            "zone_counts": dict(zone_counts),
            "unique_tracks": len(track_ids),
            "track_ids": track_ids,
        }

    def unique_values(self, field: str) -> list:
        """Return sorted unique values for a given field."""
        vals = {e.get(field) for e in self._events if e.get(field) is not None}
        return sorted(vals, key=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_list(val: str | list[str] | None) -> set[str] | None:
    if val is None:
        return None
    if isinstance(val, str):
        return {val}
    return set(val)


def _normalise_int_list(val: int | list[int] | None) -> set[int] | None:
    if val is None:
        return None
    if isinstance(val, int):
        return {val}
    return set(val)
