"""Export utilities for Exam Sheet Parser.

Supports:
- JSON: structured exam output with headings, questions, MCQ options, marks.
- CSV: one row per question with marks and option count.

Usage::

    from export import ExamSheetExporter
    from config import ExamSheetConfig

    with ExamSheetExporter(cfg) as exporter:
        exporter.write(result, report=report, source="exam.jpg")
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

from config import ExamSheetConfig
from parser import ExamSheetResult
from validator import ValidationReport

log = logging.getLogger("exam_sheet.export")

_CSV_COLUMNS = [
    "source",
    "question_number",
    "question_text",
    "marks",
    "num_options",
    "option_letters",
    "confidence",
    "valid",
]


class ExamSheetExporter:
    """Write exam sheet results to JSON and/or CSV."""

    def __init__(self, cfg: ExamSheetConfig) -> None:
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

    def __enter__(self) -> ExamSheetExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write ---------------------------------------------------------

    def write(
        self,
        result: ExamSheetResult,
        report: ValidationReport | None = None,
        source: str = "",
    ) -> None:
        """Record one exam sheet's parsed questions."""
        valid = report.valid if report else True

        # CSV: one row per question
        if self._csv_writer is not None:
            for q in result.questions:
                row = {
                    "source": source,
                    "question_number": q.number,
                    "question_text": q.text[:200],
                    "marks": q.marks if q.marks is not None else "",
                    "num_options": len(q.options),
                    "option_letters": ",".join(q.option_letters),
                    "confidence": round(q.confidence, 4),
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
                "total_sheets": len(self._json_records),
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
        result: ExamSheetResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        questions = []
        for q in result.questions:
            questions.append({
                "number": q.number,
                "text": q.text,
                "marks": q.marks,
                "options": q.options,
                "option_letters": q.option_letters,
                "body_lines": q.body_lines,
                "confidence": round(q.confidence, 4),
            })

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "headings": result.headings,
            "sections": result.sections,
            "num_blocks": result.num_blocks,
            "num_questions": result.num_questions,
            "total_marks": result.total_marks,
            "mean_confidence": round(result.mean_confidence, 4),
            "questions": questions,
        }
        if report is not None:
            record["validation_valid"] = report.valid
            record["validation_summary"] = report.summary()

        return record
