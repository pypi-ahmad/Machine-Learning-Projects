"""Per-frame metric export for Gesture Controlled Slideshow.
"""Per-frame metric export for Gesture Controlled Slideshow.

Logs gesture, action, slide state per frame to CSV/JSON.
"""
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

from config import GestureConfig
from controller import ControllerResult

log = logging.getLogger("gesture.export")

_CSV_COLUMNS = [
    "frame",
    "hand_detected",
    "gesture",
    "finger_count",
    "confidence",
    "action",
    "triggered",
    "slide_index",
    "total_slides",
    "paused",
    "pointer_mode",
]


class GestureExporter:
    """Write per-frame gesture metrics to CSV and/or JSON."""

    def __init__(self, cfg: GestureConfig) -> None:
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

    def __enter__(self) -> GestureExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, result: ControllerResult, frame_idx: int = 0) -> None:
        """Record one frame's gesture metrics."""
        row = {
            "frame": frame_idx,
            "hand_detected": result.hand_detected,
            "gesture": result.gesture.gesture,
            "finger_count": result.gesture.finger_count,
            "confidence": round(result.gesture.confidence, 3),
            "action": result.debounced.action,
            "triggered": result.debounced.triggered,
            "slide_index": result.slide.current_index,
            "total_slides": result.slide.total_slides,
            "paused": result.slide.paused,
            "pointer_mode": result.slide.pointer_mode,
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
