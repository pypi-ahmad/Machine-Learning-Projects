"""Exercise Rep Counter — per-frame metrics export (CSV / JSON)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from controller import ControllerResult


class FrameExporter:
    """Write per-frame exercise metrics to disk."""

    _CSV_FIELDS = [
        "frame",
        "pose_detected",
        "exercise",
        "angle",
        "stage",
        "reps",
        "side",
    ]

    def __init__(
        self,
        csv_path: str | Path | None = None,
        json_path: str | Path | None = None,
    ) -> None:
        self._csv_path = Path(csv_path) if csv_path else None
        self._json_path = Path(json_path) if json_path else None
        self._csv_file = None
        self._csv_writer = None
        self._json_records: list[dict[str, Any]] = []
        self._frame = 0

    def __enter__(self) -> FrameExporter:
        if self._csv_path:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._CSV_FIELDS)
            self._csv_writer.writeheader()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._csv_file:
            self._csv_file.close()
        if self._json_path and self._json_records:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            self._json_path.write_text(
                json.dumps(self._json_records, indent=2), encoding="utf-8"
            )

    def record(self, result: ControllerResult) -> None:
        self._frame += 1
        row: dict[str, Any] = {
            "frame": self._frame,
            "pose_detected": result.pose.detected,
        }
        if result.analysis and result.rep_state:
            row["exercise"] = result.analysis.exercise
            row["angle"] = round(result.analysis.angle, 1)
            row["stage"] = result.rep_state.stage
            row["reps"] = result.rep_state.reps
            row["side"] = result.analysis.side
        else:
            row.update({"exercise": "", "angle": 0, "stage": "", "reps": 0, "side": ""})

        if self._csv_writer:
            self._csv_writer.writerow(row)
        if self._json_path is not None:
            self._json_records.append(row)
