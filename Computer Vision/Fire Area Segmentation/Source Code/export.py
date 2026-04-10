"""Fire Area Segmentation — structured output export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from metrics import FrameMetrics
from segmentation import SegmentationResult
from trend import TrendSummary


# ── JSON export ────────────────────────────────────────────


def export_frame_json(
    path: str | Path,
    seg: SegmentationResult,
    metrics: FrameMetrics,
    trend: TrendSummary | None = None,
    *,
    source: str = "",
    frame_idx: int | None = None,
) -> None:
    """Write a JSON report for one image / frame."""
    record: dict[str, Any] = {
        "source": str(source),
        "fire_coverage": metrics.fire_coverage,
        "smoke_coverage": metrics.smoke_coverage,
        "fire_area_px": metrics.fire_area_px,
        "smoke_area_px": metrics.smoke_area_px,
        "fire_count": metrics.fire_count,
        "smoke_count": metrics.smoke_count,
        "mean_fire_conf": metrics.mean_fire_conf,
        "mean_smoke_conf": metrics.mean_smoke_conf,
        "instances": [
            {
                "class": inst.class_name,
                "confidence": inst.confidence,
                "area_px": inst.area_px,
                "bbox": list(inst.bbox),
            }
            for inst in seg.instances
        ],
    }
    if frame_idx is not None:
        record["frame"] = frame_idx
    if trend is not None:
        record["trend"] = {
            "window_size": trend.window_size,
            "frames_seen": trend.frames_seen,
            "avg_fire_coverage": trend.avg_fire_coverage,
            "peak_fire_coverage": trend.peak_fire_coverage,
            "fire_growth_rate": trend.fire_growth_rate,
        }
    _write_json(path, record)


# ── mask saving ────────────────────────────────────────────


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    """Save a binary mask as PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), mask)


# ── CSV exporter ───────────────────────────────────────────


class CSVExporter:
    """Append per-frame rows to a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._header_written = False

    def open(self) -> None:
        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

    def write_row(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self.open()
        if self._header_written is False:
            self._writer.writerow(list(row.keys()))
            self._header_written = True
        self._writer.writerow(list(row.values()))

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


# ── helpers ────────────────────────────────────────────────


def _write_json(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
