"""Waterbody & Flood Extent Segmentation — structured output export."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from coverage import CoverageMetrics, FloodChangeMetrics
from flood_compare import ComparisonResult
from segmentation import SegmentationResult


# ── single-image JSON ─────────────────────────────────────


def export_single_json(
    path: str | Path,
    seg: SegmentationResult,
    metrics: CoverageMetrics,
    *,
    source: str = "",
    frame_idx: int | None = None,
) -> None:
    """Write a JSON report for one image / frame."""
    record: dict[str, Any] = {
        "source": str(source),
        "coverage_ratio": metrics.coverage_ratio,
        "water_area_px": metrics.water_area_px,
        "instance_count": metrics.instance_count,
        "mean_confidence": metrics.mean_confidence,
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


# ── comparison JSON ────────────────────────────────────────


def export_comparison_json(
    path: str | Path,
    metrics: FloodChangeMetrics,
    comparison: ComparisonResult | None = None,
    *,
    before_path: str = "",
    after_path: str = "",
) -> None:
    """Write a JSON report for a before/after comparison."""
    record: dict[str, Any] = {
        "before": str(before_path),
        "after": str(after_path),
        "before_coverage": metrics.before_coverage,
        "after_coverage": metrics.after_coverage,
        "flooded_new_px": metrics.flooded_new_px,
        "receded_px": metrics.receded_px,
        "permanent_px": metrics.permanent_px,
        "net_change_ratio": metrics.net_change_ratio,
        "iou": metrics.iou,
        "num_new_regions": metrics.num_new_regions,
        "num_receded_regions": metrics.num_receded_regions,
        "regions": [
            {
                "label": r.label,
                "area_px": r.area_px,
                "bbox": list(r.bbox),
                "centroid": list(r.centroid),
            }
            for r in (comparison.regions if comparison else [])
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

    def write_row(
        self,
        row: dict[str, Any],
    ) -> None:
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
