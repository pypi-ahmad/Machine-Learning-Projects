"""Export utilities for Prescription OCR Parser.

Supports:
- JSON: full extraction with medicines, confidence, header fields.
- CSV: one row per medicine entry with dosage/frequency columns.

Usage::

    from export import PrescriptionExporter
    from config import PrescriptionConfig

    with PrescriptionExporter(cfg) as exporter:
        exporter.write(parse_result, report=report, source="rx.jpg")
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

from config import PrescriptionConfig
from parser import PrescriptionResult
from validator import ValidationReport

log = logging.getLogger("prescription_ocr.export")

_CSV_COLUMNS = [
    "source",
    "medicine_name",
    "dosage",
    "frequency",
    "duration",
    "route",
    "instructions",
    "confidence",
    "prescriber",
    "patient_name",
    "date",
    "valid",
]


class PrescriptionExporter:
    """Write extraction results to JSON and/or CSV."""

    def __init__(self, cfg: PrescriptionConfig) -> None:
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

    def __enter__(self) -> PrescriptionExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write ---------------------------------------------------------

    def write(
        self,
        result: PrescriptionResult,
        report: ValidationReport | None = None,
        source: str = "",
    ) -> None:
        """Record one prescription extraction."""
        valid = report.valid if report else True

        # CSV: one row per medicine
        if self._csv_writer is not None:
            for med in result.medicines:
                row = {
                    "source": source,
                    "medicine_name": med.medicine_name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "duration": med.duration,
                    "route": med.route,
                    "instructions": med.instructions,
                    "confidence": round(med.confidence, 3),
                    "prescriber": result.header_fields.get("prescriber", _empty()).value,
                    "patient_name": result.header_fields.get("patient_name", _empty()).value,
                    "date": result.header_fields.get("date", _empty()).value,
                    "valid": valid,
                }
                self._csv_writer.writerow(row)

        # JSON
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
                "disclaimer": (
                    "This output is for informational purposes only. "
                    "It does not constitute medical advice. Always consult "
                    "a licensed healthcare professional."
                ),
                "total_prescriptions": len(self._json_records),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "records": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info(
                "JSON export → %s (%d records)",
                out, len(self._json_records),
            )

    # -- internal ------------------------------------------------------

    def _to_json_record(
        self,
        result: PrescriptionResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        medicines = []
        for med in result.medicines:
            medicines.append({
                "medicine_name": med.medicine_name,
                "dosage": med.dosage,
                "frequency": med.frequency,
                "duration": med.duration,
                "route": med.route,
                "instructions": med.instructions,
                "confidence": round(med.confidence, 3),
            })

        header_fields = {}
        for name, ef in result.header_fields.items():
            header_fields[name] = {
                "value": ef.value,
                "confidence": round(ef.confidence, 3),
            }

        record: dict[str, Any] = {
            "source": source,
            "num_blocks": result.num_blocks,
            "mean_confidence": round(result.mean_confidence, 3),
            "header_fields": header_fields,
            "medicines": medicines,
            "num_medicines": result.num_medicines,
        }
        if report:
            record["valid"] = report.valid
            record["warnings"] = [
                {
                    "field": w.field_name,
                    "message": w.message,
                    "severity": w.severity,
                }
                for w in report.warnings
            ]
        return record


class _EmptyField:
    """Sentinel for missing header fields in CSV export."""
    value = ""


def _empty() -> _EmptyField:
    return _EmptyField()
