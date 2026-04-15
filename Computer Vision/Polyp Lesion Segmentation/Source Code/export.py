"""Polyp Lesion Segmentation -- structured output export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from metrics import PolypMetrics
from segmentation import SegmentationResult


# ── JSON export ────────────────────────────────────────────


def export_frame_json(
    path: str | Path,
    seg: SegmentationResult,
    metrics: PolypMetrics,
    *,
    source: str = "",
    frame_idx: int | None = None,
    backend: str = "yolo",
) -> None:
    """Write a JSON report for one image."""
    record: dict[str, Any] = {
        "source": str(source),
        "backend": backend,
        "polyp_area_px": metrics.polyp_area_px,
        "polyp_coverage": metrics.polyp_coverage,
        "polyp_count": metrics.polyp_count,
        "mean_confidence": metrics.mean_confidence,
        "largest_polyp_px": metrics.largest_polyp_px,
        "instances": [
            {
                "confidence": inst.confidence,
                "area_px": inst.area_px,
                "bbox": list(inst.bbox),
            }
            for inst in seg.instances
        ],
    }
    if metrics.dice is not None:
        record["dice"] = metrics.dice
        record["iou"] = metrics.iou
    if frame_idx is not None:
        record["frame"] = frame_idx
    _write_json(path, record)


def export_batch_json(
    path: str | Path,
    entries: list[dict[str, Any]],
) -> None:
    """Write a JSON report for a batch of images."""
    _write_json(path, {"images": entries, "total": len(entries)})


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


def _write_json(path: str | Path, record: dict | list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
