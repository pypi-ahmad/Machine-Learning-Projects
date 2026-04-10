"""Industrial Scratch / Crack Segmentation — structured output export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from metrics import DefectMetrics
from segmentation import SegmentationResult


# ── JSON export ────────────────────────────────────────────


def export_frame_json(
    path: str | Path,
    seg: SegmentationResult,
    metrics: DefectMetrics,
    *,
    source: str = "",
    frame_idx: int | None = None,
) -> None:
    """Write a JSON report for one image."""
    record: dict[str, Any] = {
        "source": str(source),
        "defect_count": metrics.defect_count,
        "total_defect_area_px": metrics.total_defect_area_px,
        "defect_coverage": metrics.defect_coverage,
        "severity": metrics.severity,
        "max_length_px": metrics.max_length_px,
        "mean_length_px": metrics.mean_length_px,
        "max_aspect_ratio": metrics.max_aspect_ratio,
        "mean_confidence": metrics.mean_confidence,
        "instances": [
            {
                "confidence": inst.confidence,
                "area_px": inst.area_px,
                "length_px": inst.length_px,
                "aspect_ratio": inst.aspect_ratio,
                "bbox": list(inst.bbox),
            }
            for inst in seg.instances
        ],
    }
    if frame_idx is not None:
        record["frame"] = frame_idx
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
