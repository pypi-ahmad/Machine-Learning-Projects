"""Export utilities for Waste Sorting Detector.

Supports CSV and JSON per-frame export with:
- Per-class counts
- Bin-zone validation results
- Misplaced item details
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WasteConfig
from sorter import FrameResult

log = logging.getLogger("waste_sorting.export")

CSV_FIELDS = [
    "frame_idx",
    "total_items",
    "class_counts",
    "misplaced_count",
    "misplaced_details",
]


class WasteExporter:
    """Write per-frame detection summaries to CSV and/or JSON."""

    def __init__(self, cfg: WasteConfig) -> None:
        self.cfg = cfg
        self._csv_writer: csv.DictWriter | None = None
        self._csv_fh = None
        self._json_records: list[dict[str, Any]] = []

        if cfg.export_csv:
            out = Path(cfg.export_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(out, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=CSV_FIELDS)
            self._csv_writer.writeheader()
            log.info("CSV export → %s", out)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, result: FrameResult) -> None:
        """Append one frame result to all active sinks."""
        row = self._to_row(result)

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

        if self.cfg.export_json:
            self._json_records.append(row)

    def close(self) -> None:
        """Flush and close all export sinks."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            log.info("CSV export closed")

        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self._json_records, indent=2), encoding="utf-8")
            log.info("JSON export → %s (%d records)", out, len(self._json_records))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_row(result: FrameResult) -> dict[str, Any]:
        misplaced_details = [
            {
                "class": it.class_name,
                "zone": it.zone_name,
                "bbox": list(it.bbox),
            }
            for it in result.misplaced_items
        ]
        return {
            "frame_idx": result.frame_idx,
            "total_items": result.total_items,
            "class_counts": json.dumps(result.class_counts),
            "misplaced_count": len(result.misplaced_items),
            "misplaced_details": json.dumps(misplaced_details),
        }

    def __enter__(self) -> WasteExporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
