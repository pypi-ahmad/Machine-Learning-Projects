"""Road Pothole Segmentation — structured output export.

Saves per-frame or per-image results as JSON (structured) or CSV (tabular).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from severity import SeverityReport


def export_json(
    path: str | Path,
    report: SeverityReport,
    *,
    source: str = "",
    frame_idx: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a JSON report for one image / frame."""
    record: dict[str, Any] = {
        "source": str(source),
        "road_condition": report.road_condition,
        "total_potholes": report.total_count,
        "minor": report.minor_count,
        "moderate": report.moderate_count,
        "severe": report.severe_count,
        "total_area_px": report.total_area_px,
        "total_area_m2": report.total_area_m2,
        "potholes": [
            {
                "id": a.instance_id,
                "severity": a.severity,
                "area_px": a.area_px,
                "area_m2": a.area_m2,
                "confidence": a.confidence,
                "bbox": list(a.bbox),
            }
            for a in report.assessments
        ],
    }
    if frame_idx is not None:
        record["frame"] = frame_idx
    if extra:
        record.update(extra)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


class CSVExporter:
    """Append per-frame rows to a CSV file."""

    _HEADER = [
        "frame", "source", "total_potholes",
        "minor", "moderate", "severe",
        "total_area_px", "road_condition",
    ]

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None

    def open(self) -> None:
        need_header = not self._path.exists()
        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if need_header:
            self._writer.writerow(self._HEADER)

    def write_row(
        self,
        report: SeverityReport,
        *,
        source: str = "",
        frame_idx: int = 0,
    ) -> None:
        if self._writer is None:
            self.open()
        self._writer.writerow([
            frame_idx, source, report.total_count,
            report.minor_count, report.moderate_count, report.severe_count,
            report.total_area_px, report.road_condition,
        ])

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
