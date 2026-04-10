"""Finger Counter Pro — per-frame metrics export (CSV / JSON)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from finger_counter import FingerState


class FrameExporter:
    """Write per-frame finger-counting metrics to disk."""

    _CSV_FIELDS = [
        "frame",
        "hand_count",
        "total_raw",
        "total_smoothed",
        # per-hand (up to 2 hands)
        "h0_hand",
        "h0_count_raw",
        "h0_count_smooth",
        "h0_thumb",
        "h0_index",
        "h0_middle",
        "h0_ring",
        "h0_pinky",
        "h1_hand",
        "h1_count_raw",
        "h1_count_smooth",
        "h1_thumb",
        "h1_index",
        "h1_middle",
        "h1_ring",
        "h1_pinky",
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

    # -- context manager ------------------------------------------------
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

    # -- recording -------------------------------------------------------
    def record(
        self,
        per_hand: list[FingerState],
        total_raw: int,
        smoothed_per_hand: dict[str, int],
        smoothed_total: int,
    ) -> None:
        self._frame += 1
        row: dict[str, Any] = {
            "frame": self._frame,
            "hand_count": len(per_hand),
            "total_raw": total_raw,
            "total_smoothed": smoothed_total,
        }
        for i, state in enumerate(per_hand[:2]):
            p = f"h{i}_"
            row[p + "hand"] = state.handedness
            row[p + "count_raw"] = state.finger_count
            row[p + "count_smooth"] = smoothed_per_hand.get(state.handedness, state.finger_count)
            for j, name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
                row[p + name] = int(state.fingers_up[j])

        if self._csv_writer:
            self._csv_writer.writerow(row)
        if self._json_path is not None:
            self._json_records.append(row)
