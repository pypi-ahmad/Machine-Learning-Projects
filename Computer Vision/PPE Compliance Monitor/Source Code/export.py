"""PPE Compliance Monitor — event exporter.

Logs compliance events (violations & stats) to CSV / JSON files one row
per frame, and saves violation snapshot images with a configurable
per-zone cooldown.

Usage::

    from export import EventExporter

    exporter = EventExporter(cfg)
    exporter.log_frame(frame_idx, zone_statuses, alerts, frame)
    exporter.flush()
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from config import PPEConfig
from zones import ZoneStatus, AlertEvent

log = logging.getLogger("ppe_compliance.export")

CSV_FIELDS = [
    "frame",
    "timestamp",
    "zone",
    "total_persons",
    "compliant",
    "violations",
    "missing_items",
]


class EventExporter:
    """Write compliance events to disk."""

    def __init__(self, cfg: PPEConfig) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.export_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.out_dir / "events.csv"
        self._json_path = self.out_dir / "events.json"
        self._snap_dir = self.out_dir / "violations"

        self._rows: list[dict] = []
        self._snap_cooldowns: dict[str, float] = {}

        # Write CSV header on first run
        if cfg.save_events_csv and not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    # ---- public API --------------------------------------------------------

    def log_frame(
        self,
        frame_idx: int,
        zone_statuses: Sequence[ZoneStatus],
        alerts: Sequence[AlertEvent],
        frame: np.ndarray | None = None,
    ) -> None:
        """Record one frame's compliance data."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        for zs in zone_statuses:
            missing = []
            for pc in zs.persons:
                if not pc.is_compliant:
                    missing.extend(pc.missing_items)

            row = {
                "frame": frame_idx,
                "timestamp": ts,
                "zone": zs.name,
                "total_persons": zs.person_count,
                "compliant": zs.compliant_count,
                "violations": zs.violation_count,
                "missing_items": ";".join(sorted(set(missing))) if missing else "",
            }
            self._rows.append(row)

            if self.cfg.save_events_csv:
                self._append_csv(row)

        # Violation snapshots
        if frame is not None and self.cfg.save_violation_snapshots:
            self._save_snapshots(alerts, frame)

    def flush(self) -> None:
        """Write accumulated JSON events to disk."""
        if self.cfg.save_events_json and self._rows:
            self._json_path.write_text(
                json.dumps(self._rows, indent=2), encoding="utf-8"
            )
            log.info("Exported %d event rows -> %s", len(self._rows), self._json_path)

    # ---- internal ----------------------------------------------------------

    def _append_csv(self, row: dict) -> None:
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)

    def _save_snapshots(self, alerts: Sequence[AlertEvent], frame: np.ndarray) -> None:
        now = time.monotonic()
        cooldown = self.cfg.snapshot_cooldown_sec

        for alert in alerts:
            last = self._snap_cooldowns.get(alert.zone_name, 0.0)
            if now - last < cooldown:
                continue
            self._snap_cooldowns[alert.zone_name] = now

            self._snap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{alert.zone_name}_{ts}.jpg"
            path = self._snap_dir / fname
            cv2.imwrite(str(path), frame)
            log.info("Violation snapshot -> %s", path)
