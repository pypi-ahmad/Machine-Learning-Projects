"""Video Event Search — event store.
"""Video Event Search — event store.

Append-only JSON event log with load/save and per-event schema.

Usage::

    from event_store import EventStore

    store = EventStore("outputs/events.json")
    store.add(event)
    store.add_batch(events)
    store.flush()
    all_events = store.load()
"""
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from config import Event

log = logging.getLogger("video_event_search.event_store")

CSV_FIELDS = [
    "event_type",
    "track_id",
    "class_name",
    "frame_idx",
    "timestamp_sec",
    "confidence",
    "center_x",
    "center_y",
    "zone_or_line",
    "direction",
    "dwell_seconds",
]


class EventStore:
    """JSON + CSV event persistence."""

    def __init__(self, json_path: str | Path, csv_path: str | Path | None = None) -> None:
        self.json_path = Path(json_path)
        self.csv_path = Path(csv_path) if csv_path else self.json_path.with_suffix(".csv")
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []

        if self.csv_path and not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    def add(self, event: Event) -> None:
        row = event.to_dict()
        self._events.append(row)
        self._append_csv(row)

    def add_batch(self, events: list[Event]) -> None:
        for e in events:
            self.add(e)

    def flush(self) -> None:
        """Write all accumulated events to JSON."""
        if self._events:
            self.json_path.write_text(
                json.dumps(self._events, indent=2), encoding="utf-8",
            )
            log.info("Flushed %d events -> %s", len(self._events), self.json_path)

    def load(self) -> list[dict]:
        """Load events from JSON file on disk."""
        if self.json_path.exists():
            return json.loads(self.json_path.read_text(encoding="utf-8"))
        return []

    @property
    def count(self) -> int:
        return len(self._events)

    def _append_csv(self, row: dict) -> None:
        csv_row = {k: row.get(k, row.get(f"center_{k[-1]}", "")) for k in CSV_FIELDS}
        csv_row["center_x"] = row.get("center_x", 0)
        csv_row["center_y"] = row.get("center_y", 0)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(csv_row)
