"""Export utilities for Business Card Reader.

Supports:
- JSON: full extraction result with fields and confidence.
- CSV: one row per card with contact fields as columns.

Usage::

    from export import CardExporter
    from config import CardConfig

    exporter = CardExporter(CardConfig())
    exporter.write(parse_result, report=report, source="card.jpg")
    exporter.close()
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CardConfig
from parser import ParseResult
from validator import ValidationReport

log = logging.getLogger("business_card.export")

_CSV_COLUMNS = [
    "source",
    "name",
    "title",
    "company",
    "phone",
    "email",
    "website",
    "address",
    "fields_found",
    "valid",
    "warnings_count",
]


class CardExporter:
    """Write extraction results to JSON and/or CSV."""

    def __init__(self, cfg: CardConfig) -> None:
        self.cfg = cfg
        self._csv_writer: csv.DictWriter | None = None
        self._csv_fh = None
        self._json_records: list[dict[str, Any]] = []

        if cfg.export_csv:
            out = Path(cfg.export_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(out, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=_CSV_COLUMNS)
            self._csv_writer.writeheader()
            log.info("CSV export -> %s", out)

    def write(
        self,
        result: ParseResult,
        report: ValidationReport | None = None,
        source: str = "",
    ) -> None:
        """Record one business card extraction."""
        row = self._to_row(result, report, source)

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

        if self.cfg.export_json:
            self._json_records.append(
                self._to_json_record(result, report, source)
            )

    def close(self) -> None:
        """Flush and close all sinks."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            log.info("CSV export closed")

        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_cards": len(self._json_records),
                "exported_at": datetime.now().isoformat(),
                "cards": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("JSON export -> %s (%d cards)", out, len(self._json_records))

    def __enter__(self) -> CardExporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _to_row(
        result: ParseResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        def _fval(name: str) -> str:
            ef = result.fields.get(name)
            return ef.value if ef else ""

        return {
            "source": source,
            "name": _fval("name"),
            "title": _fval("title"),
            "company": _fval("company"),
            "phone": _fval("phone"),
            "email": _fval("email"),
            "website": _fval("website"),
            "address": _fval("address"),
            "fields_found": len(result.fields),
            "valid": report.valid if report else "",
            "warnings_count": len(report.warnings) if report else 0,
        }

    @staticmethod
    def _to_json_record(
        result: ParseResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        fields_out: dict[str, Any] = {}
        for name, ef in result.fields.items():
            fields_out[name] = {
                "value": ef.value,
                "confidence": round(ef.confidence, 4),
                "source_text": ef.source_text,
            }

        record: dict[str, Any] = {
            "source": source,
            "fields": fields_out,
            "num_ocr_blocks": result.num_blocks,
            "extracted_at": datetime.now().isoformat(),
        }
        if report:
            record["validation"] = {
                "valid": report.valid,
                "fields_found": report.fields_found,
                "fields_missing": report.fields_missing,
                "warnings": [
                    {"field": w.field_name, "message": w.message, "severity": w.severity}
                    for w in report.warnings
                ],
            }
        return record
