"""Export utilities for Scene Text Reader Translator.

Supports:
- JSON: text blocks with coordinates, confidence, translations.
- CSV: one row per text region with bounding box metadata.

Usage::

    from export import SceneTextExporter
    from config import SceneTextConfig

    with SceneTextExporter(cfg) as exporter:
        exporter.write(scene_text_result, source="street.jpg")
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

from config import SceneTextConfig
from parser import SceneTextResult
from validator import ValidationReport

log = logging.getLogger("scene_text.export")

_CSV_COLUMNS = [
    "source",
    "frame_index",
    "text",
    "translated_text",
    "confidence",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "centre_x",
    "centre_y",
    "valid",
]


class SceneTextExporter:
    """Write scene text results to JSON and/or CSV."""

    def __init__(self, cfg: SceneTextConfig) -> None:
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

    # -- context manager -----------------------------------------------

    def __enter__(self) -> SceneTextExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write ---------------------------------------------------------

    def write(
        self,
        result: SceneTextResult,
        report: ValidationReport | None = None,
        source: str = "",
    ) -> None:
        """Record one image/frame's text reads."""
        valid = report.valid if report else True

        # CSV: one row per text block
        if self._csv_writer is not None:
            for read in result.reads:
                x_min = min(p[0] for p in read.bbox)
                y_min = min(p[1] for p in read.bbox)
                x_max = max(p[0] for p in read.bbox)
                y_max = max(p[1] for p in read.bbox)
                row = {
                    "source": source,
                    "frame_index": result.frame_index,
                    "text": read.text,
                    "translated_text": read.translated_text,
                    "confidence": round(read.confidence, 4),
                    "bbox_x_min": x_min,
                    "bbox_y_min": y_min,
                    "bbox_x_max": x_max,
                    "bbox_y_max": y_max,
                    "centre_x": read.centre[0],
                    "centre_y": read.centre[1],
                    "valid": valid,
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
            payload = {
                "total_images": len(self._json_records),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "records": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info(
                "JSON export -> %s (%d records)",
                out, len(self._json_records),
            )

    # -- internal ------------------------------------------------------

    @staticmethod
    def _to_json_record(
        result: SceneTextResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        blocks = []
        for read in result.reads:
            blocks.append({
                "text": read.text,
                "translated_text": read.translated_text,
                "confidence": round(read.confidence, 4),
                "bbox": read.bbox,
                "centre": list(read.centre),
            })

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "frame_index": result.frame_index,
            "num_blocks": result.num_blocks,
            "mean_confidence": round(result.mean_confidence, 4),
            "raw_text": result.raw_text,
            "translation_enabled": result.translation_enabled,
            "translation_provider": result.translation_provider,
            "blocks": blocks,
        }
        if report is not None:
            record["validation_valid"] = report.valid
            record["validation_summary"] = report.summary()

        return record
