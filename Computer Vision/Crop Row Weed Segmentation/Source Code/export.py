"""Crop Row & Weed Segmentation — structured output export.

Saves per-frame results as JSON (per-instance details + class stats)
or CSV (per-frame summary row).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from class_stats import AreaReport
from segmentation import SegmentationResult


def export_json(
    path: str | Path,
    seg: SegmentationResult,
    report: AreaReport,
    *,
    source: str = "",
    frame_idx: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a JSON report for one image / frame."""
    record: dict[str, Any] = {
        "source": str(source),
        "total_instances": report.total_instances,
        "total_segmented_px": report.total_segmented_px,
        "background_ratio": report.background_ratio,
        "per_class": {
            name: {
                "instance_count": s.instance_count,
                "total_area_px": s.total_area_px,
                "coverage_ratio": s.coverage_ratio,
                "mean_confidence": s.mean_confidence,
            }
            for name, s in report.per_class.items()
        },
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
    if extra:
        record.update(extra)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def save_masks(
    out_dir: str | Path,
    seg: SegmentationResult,
    stem: str,
) -> None:
    """Save per-class binary masks as PNG files."""
    d = Path(out_dir) / "masks"
    d.mkdir(parents=True, exist_ok=True)
    for name, mask in seg.class_masks.items():
        cv2.imwrite(str(d / f"{stem}_{name}.png"), mask)


class CSVExporter:
    """Append per-frame rows to a CSV file."""

    _HEADER = [
        "frame", "source", "total_instances", "background_ratio",
    ]

    def __init__(self, path: str | Path, class_names: list[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._class_names = class_names
        self._file = None
        self._writer = None

    def open(self) -> None:
        need_header = not self._path.exists()
        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if need_header:
            extra_cols = []
            for cn in self._class_names:
                extra_cols += [f"{cn}_count", f"{cn}_area_px", f"{cn}_coverage"]
            self._writer.writerow(self._HEADER + extra_cols)

    def write_row(
        self,
        report: AreaReport,
        *,
        source: str = "",
        frame_idx: int = 0,
    ) -> None:
        if self._writer is None:
            self.open()
        row: list[Any] = [
            frame_idx, source, report.total_instances, report.background_ratio,
        ]
        for cn in self._class_names:
            s = report.per_class.get(cn)
            if s:
                row += [s.instance_count, s.total_area_px, s.coverage_ratio]
            else:
                row += [0, 0, 0.0]
        self._writer.writerow(row)

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
