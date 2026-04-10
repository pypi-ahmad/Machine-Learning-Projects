"""Export utilities for Crowd Zone Counter.

Supports:
- JSON: per-frame zone counts + alert log + session summary.
- CSV: per-frame row with per-zone columns.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CrowdConfig
from zone_counter import Alert, FrameResult

log = logging.getLogger("crowd_zone.export")


class CrowdExporter:
    """Write per-frame zone counts and alerts to CSV / JSON."""

    def __init__(self, cfg: CrowdConfig) -> None:
        self.cfg = cfg
        self._zone_names: list[str] = [z.name for z in cfg.zones]

        self._csv_writer: csv.DictWriter | None = None
        self._csv_fh = None
        self._json_frames: list[dict[str, Any]] = []
        self._all_alerts: list[dict[str, Any]] = []

        if cfg.export_csv:
            fields = ["frame_idx", "total_persons", "unzoned"] + self._zone_names
            out = Path(cfg.export_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(out, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=fields)
            self._csv_writer.writeheader()
            log.info("CSV export → %s", out)

    def write(self, result: FrameResult) -> None:
        """Record one frame."""
        row: dict[str, Any] = {
            "frame_idx": result.frame_idx,
            "total_persons": result.total_persons,
            "unzoned": result.unzoned_count,
        }
        for zs in result.zone_states:
            row[zs.name] = zs.count

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

        if self.cfg.export_json:
            zone_counts = {zs.name: zs.count for zs in result.zone_states}
            overcrowded = [zs.name for zs in result.zone_states if zs.overcrowded]
            self._json_frames.append({
                "frame_idx": result.frame_idx,
                "total_persons": result.total_persons,
                "unzoned": result.unzoned_count,
                "zone_counts": zone_counts,
                "overcrowded_zones": overcrowded,
            })

        for alert in result.alerts:
            self._all_alerts.append({
                "frame_idx": alert.frame_idx,
                "zone": alert.zone_name,
                "count": alert.count,
                "max_capacity": alert.max_capacity,
            })

    def close(self) -> None:
        """Flush and close all sinks."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            log.info("CSV export closed")

        if self.cfg.export_json and self._json_frames:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_frames": len(self._json_frames),
                "zones": [
                    {"name": z.name, "max_capacity": z.max_capacity}
                    for z in self.cfg.zones
                ],
                "alerts": self._all_alerts,
                "timeline": self._json_frames,
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("JSON export → %s (%d frames, %d alerts)",
                     out, len(self._json_frames), len(self._all_alerts))

    def __enter__(self) -> CrowdExporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
