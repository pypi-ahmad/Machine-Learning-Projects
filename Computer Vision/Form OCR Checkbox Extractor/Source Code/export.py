"""Export utilities for Form OCR Checkbox Extractor.

Supports:
- JSON: full extraction result with text fields, checkbox states,
  and confidence scores.
- CSV: one row per form with checkbox summary and text fields.

Usage::

    from export import FormExporter
    from config import FormCheckboxConfig

    with FormExporter(cfg) as exporter:
        exporter.write(parse_result, report=report, source="form.jpg")
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

from config import FormCheckboxConfig
from parser import FormParseResult
from validator import ValidationReport

log = logging.getLogger("form_checkbox.export")

_CSV_COLUMNS = [
    "source",
    "ocr_blocks",
    "checkboxes",
    "checked",
    "name",
    "date",
    "address",
    "phone",
    "email",
    "id_number",
    "valid",
    "warnings_count",
]


class FormExporter:
    """Write extraction results to JSON and/or CSV."""

    def __init__(self, cfg: FormCheckboxConfig) -> None:
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

    def __enter__(self) -> FormExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write ---------------------------------------------------------

    def write(
        self,
        result: FormParseResult,
        report: ValidationReport | None = None,
        source: str = "",
    ) -> None:
        """Record one form extraction."""
        row = self._to_row(result, report, source)

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

        if self.cfg.export_json:
            self._json_records.append(
                self._to_json_record(result, report, source),
            )

    # -- close ---------------------------------------------------------

    def close(self) -> None:
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            log.info("CSV export closed")

        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_forms": len(self._json_records),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "records": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("JSON export -> %s (%d records)", out, len(self._json_records))

    # -- internal ------------------------------------------------------

    def _to_row(
        self,
        result: FormParseResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source": source,
            "ocr_blocks": result.num_ocr_blocks,
            "checkboxes": result.num_checkboxes,
            "checked": result.num_checked,
            "valid": report.valid if report else True,
            "warnings_count": len(report.warnings) if report else 0,
        }
        for fname in ("name", "date", "address", "phone", "email", "id_number"):
            tf = result.text_fields.get(fname)
            row[fname] = tf.value if tf else ""
        return row

    def _to_json_record(
        self,
        result: FormParseResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        checkboxes = []
        for cb in result.checkbox_fields:
            checkboxes.append({
                "label": cb.label,
                "state": cb.state,
                "type": cb.control_type,
                "confidence": round(cb.confidence, 3),
                "fill_ratio": round(cb.fill_ratio, 3),
                "bbox": list(cb.bbox),
            })

        text_fields = {}
        for fname, tf in result.text_fields.items():
            text_fields[fname] = {
                "value": tf.value,
                "confidence": round(tf.confidence, 3),
            }

        record: dict[str, Any] = {
            "source": source,
            "ocr_blocks": result.num_ocr_blocks,
            "text_fields": text_fields,
            "checkboxes": checkboxes,
            "num_checkboxes": result.num_checkboxes,
            "num_checked": result.num_checked,
        }
        if report:
            record["valid"] = report.valid
            record["warnings"] = [
                {"field": w.field_name, "message": w.message, "severity": w.severity}
                for w in report.warnings
            ]
        return record
