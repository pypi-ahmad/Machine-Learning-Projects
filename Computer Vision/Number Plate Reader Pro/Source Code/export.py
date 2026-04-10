"""Export utilities for Number Plate Reader Pro.

Supports:
- JSON: plate events with timestamps, confidence, dedup status.
- CSV: one row per plate read with detection/OCR metadata.

Usage::

    from export import PlateExporter
    from config import PlateConfig

    with PlateExporter(cfg) as exporter:
        exporter.write(plate_read_result, source="frame_001")
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

from config import PlateConfig
from parser import PlateReadResult
from validator import ValidationReport

log = logging.getLogger("plate_reader.export")

_CSV_COLUMNS = [
    "source",
    "frame_index",
    "plate_text",
    "raw_text",
    "det_confidence",
    "ocr_confidence",
    "is_new",
    "is_valid",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "valid_report",
]


class PlateExporter:
    """Write plate read results to JSON and/or CSV."""

    def __init__(self, cfg: PlateConfig) -> None:
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

    # -- context manager -----------------------------------------------

    def __enter__(self) -> PlateExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write ---------------------------------------------------------

    def write(
        self,
        result: PlateReadResult,
        report: ValidationReport | None = None,
        source: str = "",
    ) -> None:
        """Record one frame's plate reads."""
        valid = report.valid if report else True

        # CSV: one row per plate read
        if self._csv_writer is not None:
            for read in result.reads:
                x1, y1, x2, y2 = read.box
                row = {
                    "source": source,
                    "frame_index": result.frame_index,
                    "plate_text": read.plate_text,
                    "raw_text": read.raw_text,
                    "det_confidence": round(read.det_confidence, 4),
                    "ocr_confidence": round(read.ocr_confidence, 4),
                    "is_new": read.is_new,
                    "is_valid": read.is_valid,
                    "box_x1": x1,
                    "box_y1": y1,
                    "box_x2": x2,
                    "box_y2": y2,
                    "valid_report": valid,
                }
                self._csv_writer.writerow(row)

        # JSON: accumulate records
        if self.cfg.export_json:
            self._json_records.append(
                self._to_json_record(result, report, source),
            )

    # -- close ---------------------------------------------------------

    def close(self) -> None:
        """Flush and close open file handles."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            log.info("CSV export closed")

        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(self._json_records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("JSON export → %s (%d records)", out, len(self._json_records))

    # -- internal ------------------------------------------------------

    @staticmethod
    def _to_json_record(
        result: PlateReadResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        plates = []
        for read in result.reads:
            plates.append({
                "plate_text": read.plate_text,
                "raw_text": read.raw_text,
                "det_confidence": round(read.det_confidence, 4),
                "ocr_confidence": round(read.ocr_confidence, 4),
                "is_new": read.is_new,
                "is_valid": read.is_valid,
                "box": list(read.box),
            })

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "frame_index": result.frame_index,
            "num_detections": result.num_detections,
            "num_valid": result.num_valid,
            "num_new": result.num_new,
            "plates": plates,
        }
        if report is not None:
            record["validation_valid"] = report.valid
            record["validation_summary"] = report.summary()

        return record
