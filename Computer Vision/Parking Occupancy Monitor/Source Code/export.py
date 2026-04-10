"""Parking Occupancy Monitor — event exporter.

Logs per-frame occupancy summaries to CSV / JSON files.

Usage::

    from export import EventExporter

    exporter = EventExporter(cfg)
    exporter.log_frame(frame_idx, frame_result)
    exporter.flush()
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from config import ParkingConfig
from slots import FrameResult

log = logging.getLogger("parking_occupancy.export")

CSV_FIELDS = [
    "frame",
    "timestamp",
    "total_slots",
    "occupied",
    "free",
    "vehicles_detected",
    "slot_details",
]


class EventExporter:
    """Write occupancy events to disk."""

    def __init__(self, cfg: ParkingConfig) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.export_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.out_dir / "occupancy.csv"
        self._json_path = self.out_dir / "occupancy.json"

        self._rows: list[dict] = []

        # Write CSV header on first run
        if cfg.save_events_csv and not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    # ---- public API --------------------------------------------------------

    def log_frame(self, frame_idx: int, result: FrameResult) -> None:
        """Record one frame's occupancy data."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        slot_details = ";".join(
            f"{ss.name}={'OCC' if ss.occupied else 'FREE'}"
            for ss in result.slot_statuses
        )

        row = {
            "frame": frame_idx,
            "timestamp": ts,
            "total_slots": result.total_slots,
            "occupied": result.occupied_count,
            "free": result.free_count,
            "vehicles_detected": len(result.vehicle_detections),
            "slot_details": slot_details,
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
            log.info("Exported %d occupancy rows → %s", len(self._rows), self._json_path)

    # ---- internal ----------------------------------------------------------

    def _append_csv(self, row: dict) -> None:
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)
