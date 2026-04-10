"""Traffic Violation Analyzer — event exporter.

Logs violation / counting events to CSV and JSON, one row per event.

Usage::

    from export import EventExporter

    exporter = EventExporter(cfg)
    exporter.log_events(frame_events)
    exporter.flush()
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from config import TrafficConfig
from rules import FrameEvents, ViolationEvent

log = logging.getLogger("traffic_violation.export")

CSV_FIELDS = [
    "frame",
    "timestamp",
    "event_type",
    "line_or_zone",
    "track_id",
    "class_name",
    "direction",
    "confidence",
    "center_x",
    "center_y",
]


class EventExporter:
    """Write traffic events to disk."""

    def __init__(self, cfg: TrafficConfig) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.export_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.out_dir / "events.csv"
        self._json_path = self.out_dir / "events.json"

        self._rows: list[dict] = []

        # Write CSV header on first run
        if cfg.save_events_csv and not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    # ---- public API --------------------------------------------------------

    def log_events(self, fe: FrameEvents) -> None:
        """Record events from one frame."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        for evt in fe.events:
            row = {
                "frame": evt.frame_idx,
                "timestamp": ts,
                "event_type": evt.event_type,
                "line_or_zone": evt.line_or_zone,
                "track_id": evt.track_id,
                "class_name": evt.class_name,
                "direction": evt.direction,
                "confidence": f"{evt.confidence:.3f}",
                "center_x": evt.center[0],
                "center_y": evt.center[1],
            }
            self._rows.append(row)

            if self.cfg.save_events_csv:
                self._append_csv(row)

    def flush(self) -> None:
        """Write accumulated JSON events to disk."""
        if self.cfg.save_events_json and self._rows:
            self._json_path.write_text(
                json.dumps(self._rows, indent=2), encoding="utf-8"
            )
            log.info("Exported %d events → %s", len(self._rows), self._json_path)

    @property
    def event_count(self) -> int:
        return len(self._rows)

    # ---- internal ----------------------------------------------------------

    def _append_csv(self, row: dict) -> None:
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)
