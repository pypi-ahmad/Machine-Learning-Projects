"""Export per-frame metrics for Blink Headpose Analyzer.

Supports CSV and JSON export of per-frame EAR, blink, and
head-pose metrics.

Usage::

    from export import FrameExporter

    with FrameExporter(cfg) as exp:
        exp.write(result, frame_idx=0)
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

from analyzer import AnalysisResult
from config import AnalyzerConfig

log = logging.getLogger("blink_headpose.export")

_CSV_COLUMNS = [
    "frame",
    "face_detected",
    "ear",
    "left_ear",
    "right_ear",
    "eyes_closed",
    "blink_detected",
    "total_blinks",
    "yaw",
    "pitch",
    "roll",
    "off_center",
]


class FrameExporter:
    """Write per-frame metrics to CSV and/or JSON."""

    def __init__(self, cfg: AnalyzerConfig) -> None:
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

    def __enter__(self) -> FrameExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, result: AnalysisResult, frame_idx: int = 0) -> None:
        """Record one frame's metrics."""
        row = {
            "frame": frame_idx,
            "face_detected": result.face_detected,
            "ear": round(result.blink.ear, 4),
            "left_ear": round(result.blink.left_ear, 4),
            "right_ear": round(result.blink.right_ear, 4),
            "eyes_closed": result.blink.eyes_closed,
            "blink_detected": result.blink.blink_detected,
            "total_blinks": result.blink.total_blinks,
            "yaw": round(result.head_pose.yaw, 1),
            "pitch": round(result.head_pose.pitch, 1),
            "roll": round(result.head_pose.roll, 1),
            "off_center": result.head_pose.off_center,
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
            log.info("JSON export → %s", out)
