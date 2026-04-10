"""Conveyor Part Defect Detector — event exporter.

Logs per-frame inspection results (pass/fail, defect details) to
CSV / JSON files and saves cropped defect thumbnails.

Usage::

    from export import EventExporter

    exporter = EventExporter(cfg)
    exporter.log_frame(frame_idx, frame_result, frame)
    exporter.flush()
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from config import InspectionConfig
from inspector import FrameResult

log = logging.getLogger("conveyor_defect.export")

CSV_FIELDS = [
    "frame",
    "timestamp",
    "verdict",
    "defect_count",
    "defect_classes",
    "total_detections",
]


class EventExporter:
    """Write inspection events and defect crops to disk."""

    def __init__(self, cfg: InspectionConfig) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.export_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.out_dir / "inspection_log.csv"
        self._json_path = self.out_dir / "inspection_log.json"
        self._crop_dir = self.out_dir / "defect_crops"

        self._rows: list[dict] = []
        self._crop_counter = 0

        # Write CSV header on first run
        if cfg.save_events_csv and not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    # ---- public API --------------------------------------------------------

    def log_frame(
        self,
        frame_idx: int,
        result: FrameResult,
        frame: np.ndarray | None = None,
    ) -> None:
        """Record one frame's inspection result."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        defect_classes = ";".join(sorted({d.class_name for d in result.defects})) if result.defects else ""

        row = {
            "frame": frame_idx,
            "timestamp": ts,
            "verdict": result.verdict,
            "defect_count": result.defect_count,
            "defect_classes": defect_classes,
            "total_detections": len(result.all_detections),
        }
        self._rows.append(row)

        if self.cfg.save_events_csv:
            self._append_csv(row)

        # Save defect crops
        if frame is not None and self.cfg.save_crops and result.defects:
            self._save_crops(result, frame)

    def flush(self) -> None:
        """Write accumulated JSON events to disk."""
        if self.cfg.save_events_json and self._rows:
            self._json_path.write_text(
                json.dumps(self._rows, indent=2), encoding="utf-8"
            )
            log.info("Exported %d inspection rows → %s", len(self._rows), self._json_path)

    # ---- internal ----------------------------------------------------------

    def _append_csv(self, row: dict) -> None:
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)

    def _save_crops(self, result: FrameResult, frame: np.ndarray) -> None:
        self._crop_dir.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        pad = self.cfg.crop_padding

        for det in result.defects:
            x1 = max(0, det.box[0] - pad)
            y1 = max(0, det.box[1] - pad)
            x2 = min(w, det.box[2] + pad)
            y2 = min(h, det.box[3] + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = f"defect_{self._crop_counter:05d}_{det.class_name}.jpg"
            cv2.imwrite(str(self._crop_dir / fname), crop)
            self._crop_counter += 1
