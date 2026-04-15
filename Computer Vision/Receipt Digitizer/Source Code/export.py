"""Export utilities for Receipt Digitizer.

Supports:
- JSON: full extraction result with fields, confidence, line items.
- CSV: one row per receipt with key fields as columns.

Usage::

    from export import ReceiptExporter
    from config import ReceiptConfig

    exporter = ReceiptExporter(ReceiptConfig())
    exporter.write(parse_result, report=report, source="receipt.jpg")
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

from config import ReceiptConfig
from parser import ParseResult
from validator import ValidationReport

log = logging.getLogger("receipt_digitizer.export")

_CSV_COLUMNS = [
    "source",
    "merchant_name",
    "date",
    "time",
    "subtotal",
    "tax",
    "tip",
    "total",
    "currency",
    "payment_method",
    "line_items_count",
    "valid",
    "warnings_count",
]


class ReceiptExporter:
    """Write extraction results to JSON and/or CSV."""

    def __init__(self, cfg: ReceiptConfig) -> None:
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
        """Record one receipt extraction."""
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
                "total_receipts": len(self._json_records),
                "exported_at": datetime.now().isoformat(),
                "receipts": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("JSON export -> %s (%d receipts)", out, len(self._json_records))

    def __enter__(self) -> ReceiptExporter:
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
            "merchant_name": _fval("merchant_name"),
            "date": _fval("date"),
            "time": _fval("time"),
            "subtotal": _fval("subtotal"),
            "tax": _fval("tax"),
            "tip": _fval("tip"),
            "total": _fval("total"),
            "currency": _fval("currency"),
            "payment_method": _fval("payment_method"),
            "line_items_count": len(result.line_items),
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

        line_items_out = [
            {
                "description": li.description,
                "quantity": li.quantity,
                "unit_price": li.unit_price,
                "amount": li.amount,
                "confidence": round(li.confidence, 4),
            }
            for li in result.line_items
        ]

        record: dict[str, Any] = {
            "source": source,
            "fields": fields_out,
            "line_items": line_items_out,
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
