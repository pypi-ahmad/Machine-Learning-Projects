"""Export utilities for Face Verification Attendance System.

Supports JSON and CSV export of attendance results.
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

from config import FaceAttendanceConfig
from parser import AttendanceResult

log = logging.getLogger("face_attendance.export")

_CSV_COLUMNS = [
    "source",
    "identity",
    "similarity",
    "matched",
    "det_confidence",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
]


class AttendanceExporter:
    """Write attendance results to JSON and/or CSV."""

    def __init__(self, cfg: FaceAttendanceConfig) -> None:
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

    def __enter__(self) -> AttendanceExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(
        self,
        result: AttendanceResult,
        source: str = "",
    ) -> None:
        """Record one frame's attendance results."""
        # CSV: one row per matched face
        if self._csv_writer is not None:
            for m in result.matches:
                row = {
                    "source": source,
                    "identity": m.identity,
                    "similarity": round(m.similarity, 4),
                    "matched": m.matched,
                    "det_confidence": round(m.det_confidence, 4),
                    "box_x1": m.box[0],
                    "box_y1": m.box[1],
                    "box_x2": m.box[2],
                    "box_y2": m.box[3],
                }
                self._csv_writer.writerow(row)

        # JSON: accumulate
        if self.cfg.export_json:
            self._json_records.append(
                self._to_json_record(result, source),
            )

    def close(self) -> None:
        """Flush and close file handles."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None

        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_frames": len(self._json_records),
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

    @staticmethod
    def _to_json_record(
        result: AttendanceResult,
        source: str,
    ) -> dict[str, Any]:
        faces = []
        for m in result.matches:
            faces.append({
                "identity": m.identity,
                "similarity": round(m.similarity, 4),
                "matched": m.matched,
                "det_confidence": round(m.det_confidence, 4),
                "box": list(m.box),
            })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "num_faces": result.num_faces,
            "num_matched": result.num_matched,
            "num_unknown": result.num_unknown,
            "backend": result.backend,
            "faces": faces,
        }
