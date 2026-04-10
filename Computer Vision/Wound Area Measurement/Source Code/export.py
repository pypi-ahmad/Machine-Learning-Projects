"""Wound Area Measurement — structured output export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from change_tracker import ChangeSummary
from metrics import WoundMetrics
from segmentation import SegmentationResult


# ── JSON export ────────────────────────────────────────────


def export_frame_json(
    path: str | Path,
    seg: SegmentationResult,
    metrics: WoundMetrics,
    *,
    source: str = "",
    frame_idx: int | None = None,
) -> None:
    """Write a JSON report for one image."""
    record: dict[str, Any] = {
        "source": str(source),
        "wound_area_px": metrics.wound_area_px,
        "wound_coverage": metrics.wound_coverage,
        "wound_count": metrics.wound_count,
        "mean_confidence": metrics.mean_confidence,
        "largest_wound_px": metrics.largest_wound_px,
        "instances": [
            {
                "confidence": inst.confidence,
                "area_px": inst.area_px,
                "bbox": list(inst.bbox),
            }
            for inst in seg.instances
        ],
    }
    if frame_idx is not None:
        record["frame"] = frame_idx
    _write_json(path, record)


def export_series_json(
    path: str | Path,
    summary: ChangeSummary,
) -> None:
    """Write a JSON report for a multi-image series."""
    record: dict[str, Any] = {
        "total_images": summary.total_images,
        "initial_area_px": summary.initial_area_px,
        "final_area_px": summary.final_area_px,
        "net_change_px": summary.net_change_px,
        "net_change_ratio": summary.net_change_ratio,
        "peak_area_px": summary.peak_area_px,
        "entries": [
            {
                "index": e.index,
                "source": e.source,
                "wound_area_px": e.wound_area_px,
                "wound_coverage": e.wound_coverage,
                "delta_px": e.delta_px,
                "delta_ratio": e.delta_ratio,
            }
            for e in summary.entries
        ],
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
    """Append per-image rows to a CSV file."""

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
