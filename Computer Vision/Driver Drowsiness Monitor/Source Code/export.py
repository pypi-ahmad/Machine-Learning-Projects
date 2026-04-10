"""Export utilities for Driver Drowsiness Monitor.

Supports:
- JSON: per-frame drowsiness metrics.
- CSV: one row per frame with EAR, MAR, pose, alerts.

Usage::

    from export import DrowsinessExporter

    with DrowsinessExporter(cfg) as exporter:
        exporter.write(result, frame_idx=0)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DrowsinessConfig
from parser import DrowsinessResult

log = logging.getLogger("drowsiness.export")

_CSV_COLUMNS = [
    "frame",
    "face_detected",
    "ear",
    "eyes_closed",
    "total_blinks",
    "perclos",
    "mar",
    "mouth_open",
    "total_yawns",
    "yaw",
    "pitch",
    "distracted",
    "active_alerts",
]


class DrowsinessExporter:
    """Write drowsiness metrics to JSON and/or CSV."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self._csv_writer: csv.DictWriter | None = None
        self._csv_fh = None
        self._json_records: list[dict[str, Any]] = []

        if cfg.export_csv:
            out = Path(cfg.export_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(out, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_fh, fieldnames=_CSV_COLUMNS,
            )
            self._csv_writer.writeheader()
            log.info("CSV export → %s", out)

    def __enter__(self) -> DrowsinessExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(
        self,
        result: DrowsinessResult,
        frame_idx: int = 0,
    ) -> None:
        """Record one frame's drowsiness metrics."""
        row = {
            "frame": frame_idx,
            "face_detected": result.face_detected,
            "ear": round(result.blink.ear, 4),
            "eyes_closed": result.blink.eyes_closed,
            "total_blinks": result.blink.total_blinks,
            "perclos": round(result.blink.perclos, 4),
            "mar": round(result.yawn.mar, 4),
            "mouth_open": result.yawn.mouth_open,
            "total_yawns": result.yawn.total_yawns,
            "yaw": round(result.head_pose.yaw, 1),
            "pitch": round(result.head_pose.pitch, 1),
            "distracted": result.head_pose.distracted,
            "active_alerts": ",".join(sorted(result.active_alerts)),
        }

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

        if self.cfg.export_json:
            self._json_records.append(row)

    def close(self) -> None:
        """Flush and close file handles."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None

        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_frames": len(self._json_records),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "records": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("JSON export → %s (%d frames)", out, len(self._json_records))
