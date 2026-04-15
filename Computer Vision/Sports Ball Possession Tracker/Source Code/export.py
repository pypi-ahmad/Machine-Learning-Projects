"""Export utilities for Sports Ball Possession Tracker.

Supports:
- JSON: full possession timeline + summary.
- CSV: per-frame possession log.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PossessionConfig
from possession import PossessionEstimator, PossessionState

log = logging.getLogger("sports_possession.export")

CSV_FIELDS = [
    "frame_idx",
    "ball_detected",
    "holder_id",
    "holder_name",
    "distance_px",
    "num_players",
]


class PossessionExporter:
    """Write per-frame possession data and final summary."""

    def __init__(self, cfg: PossessionConfig) -> None:
        self.cfg = cfg
        self._csv_writer: csv.DictWriter | None = None
        self._csv_fh = None
        self._json_timeline: list[dict[str, Any]] = []

        if cfg.export_csv:
            out = Path(cfg.export_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(out, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=CSV_FIELDS)
            self._csv_writer.writeheader()
            log.info("CSV export -> %s", out)

    def write_frame(self, state: PossessionState, num_players: int) -> None:
        """Record one frame."""
        row = {
            "frame_idx": state.frame_idx,
            "ball_detected": state.ball_detected,
            "holder_id": state.current_holder_id,
            "holder_name": state.current_holder_name or "",
            "distance_px": round(state.distance_to_holder, 1) if state.distance_to_holder else "",
            "num_players": num_players,
        }

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

        if self.cfg.export_json:
            self._json_timeline.append(row)

    def close(self, estimator: PossessionEstimator) -> None:
        """Flush and close all export sinks, appending final summary."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
            log.info("CSV export closed")

        if self.cfg.export_json:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "summary": estimator.summary(),
                "timeline": self._json_timeline,
            }
            out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            log.info("JSON export -> %s (%d frames)", out, len(self._json_timeline))

    def __enter__(self) -> PossessionExporter:
        return self

    def __exit__(self, *exc: object) -> None:
        # close() needs estimator — caller must call close() explicitly
        if self._csv_fh is not None:
            self._csv_fh.close()
