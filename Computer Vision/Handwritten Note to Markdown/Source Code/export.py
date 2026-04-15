"""Export utilities for Handwritten Note to Markdown.

Supports:
- Markdown (``.md``): structured output with headers/lists
- Plain text (``.txt``): clean text without formatting
- JSON: full extraction with per-line confidence scores

Usage::

    from export import NoteExporter
    from config import NoteConfig

    with NoteExporter(cfg) as exporter:
        exporter.write(parse_result, source="note.jpg")
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import NoteConfig
from parser import NoteParseResult
from validator import ValidationReport

log = logging.getLogger("handwritten_note.export")


class NoteExporter:
    """Write recognised text to Markdown, plain text, and/or JSON."""

    def __init__(self, cfg: NoteConfig) -> None:
        self.cfg = cfg
        self._json_records: list[dict[str, Any]] = []

    # -- context manager -----------------------------------------------

    def __enter__(self) -> NoteExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write ---------------------------------------------------------

    def write(
        self,
        result: NoteParseResult,
        report: ValidationReport | None = None,
        source: str = "",
    ) -> None:
        """Export one note's results."""
        # Markdown
        if self.cfg.export_md:
            self._write_md(result, source)

        # Plain text
        if self.cfg.export_txt:
            self._write_txt(result, source)

        # JSON (batched — flushed on close)
        if self.cfg.export_json:
            self._json_records.append(
                self._to_json_record(result, report, source),
            )

    # -- close ---------------------------------------------------------

    def close(self) -> None:
        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_notes": len(self._json_records),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "records": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info(
                "JSON export -> %s (%d records)", out, len(self._json_records),
            )

    # -- internal ------------------------------------------------------

    def _write_md(self, result: NoteParseResult, source: str) -> None:
        out = Path(self.cfg.export_md)
        # For batch: append per-image sections
        out.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if out.exists() else "w"
        with open(out, mode, encoding="utf-8") as f:
            if source:
                f.write(f"\n<!-- source: {source} -->\n\n")
            f.write(result.confidence_markdown if self.cfg.show_confidence else result.markdown)
            f.write("\n---\n")
        log.info("Markdown export -> %s", out)

    def _write_txt(self, result: NoteParseResult, source: str) -> None:
        out = Path(self.cfg.export_txt)
        out.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if out.exists() else "w"
        with open(out, mode, encoding="utf-8") as f:
            if source:
                f.write(f"# {source}\n")
            f.write(result.plain_text)
            f.write("\n")
        log.info("Text export -> %s", out)

    def _to_json_record(
        self,
        result: NoteParseResult,
        report: ValidationReport | None,
        source: str,
    ) -> dict[str, Any]:
        lines = []
        for ln in result.lines:
            lines.append({
                "text": ln.text,
                "confidence": round(ln.confidence, 3),
                "y_start": ln.y_start,
                "y_end": ln.y_end,
                "height": ln.height,
                "x_offset": ln.x_offset,
            })

        record: dict[str, Any] = {
            "source": source,
            "num_lines": result.num_lines,
            "mean_confidence": round(result.mean_confidence, 3),
            "markdown": result.markdown,
            "plain_text": result.plain_text,
            "lines": lines,
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
