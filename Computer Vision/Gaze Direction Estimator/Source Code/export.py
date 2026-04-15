"""Per-frame metric export for Gaze Direction Estimator.

Supports CSV and JSON export of per-frame iris ratios,
classified gaze direction, and smoothed outputs.

Usage::

    from export import GazeExporter

    with GazeExporter(cfg) as exp:
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

from analyzer import GazeAnalysisResult
from config import GazeConfig

log = logging.getLogger("gaze.export")

_CSV_COLUMNS = [
    "frame",
    "face_detected",
    "iris_detected",
    "h_ratio",
    "v_ratio",
    "raw_direction",
    "confidence",
    "smoothed_h",
    "smoothed_v",
    "smoothed_direction",
]


class GazeExporter:
    """Write per-frame gaze metrics to CSV and/or JSON."""

    def __init__(self, cfg: GazeConfig) -> None:
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
            log.info("CSV export -> %s", out)

    def __enter__(self) -> GazeExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, result: GazeAnalysisResult, frame_idx: int = 0) -> None:
        """Record one frame's gaze metrics."""
        row = {
            "frame": frame_idx,
            "face_detected": result.face_detected,
            "iris_detected": result.iris.detected,
            "h_ratio": round(result.raw_gaze.h_ratio, 4),
            "v_ratio": round(result.raw_gaze.v_ratio, 4),
            "raw_direction": result.raw_gaze.direction,
            "confidence": result.raw_gaze.confidence,
            "smoothed_h": round(result.smoothed.smoothed_h, 4),
            "smoothed_v": round(result.smoothed.smoothed_v, 4),
            "smoothed_direction": result.direction,
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
            log.info("JSON export -> %s", out)
