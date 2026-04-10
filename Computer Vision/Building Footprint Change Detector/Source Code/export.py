"""Building Footprint Change Detector — structured output export.

Saves per-pair results as JSON (structured) or CSV (tabular).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from diff_engine import DiffResult
from metrics import ChangeMetrics


def export_json(
    path: str | Path,
    m: ChangeMetrics,
    diff: DiffResult,
    *,
    before_path: str = "",
    after_path: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a JSON report for one image pair."""
    record: dict[str, Any] = {
        "before": str(before_path),
        "after": str(after_path),
        "metrics": asdict(m),
        "regions": [
            {
                "label": r.label,
                "area_px": r.area_px,
                "bbox": list(r.bbox),
                "centroid": list(r.centroid),
            }
            for r in diff.regions
        ],
    }
    # Remove numpy masks from metrics (not JSON-serialisable)
    if extra:
        record.update(extra)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def export_csv_row(
    path: str | Path,
    m: ChangeMetrics,
    *,
    before_path: str = "",
    after_path: str = "",
    write_header: bool = False,
) -> None:
    """Append one row to a CSV file (creates if absent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "before", "after",
        "before_area_px", "after_area_px",
        "new_area_px", "demolished_area_px", "unchanged_area_px",
        "iou", "change_ratio", "growth_ratio",
        "num_new_regions", "num_demolished_regions",
    ]
    row = [
        str(before_path), str(after_path),
        m.before_area_px, m.after_area_px,
        m.new_area_px, m.demolished_area_px, m.unchanged_area_px,
        m.iou, m.change_ratio, m.growth_ratio,
        m.num_new_regions, m.num_demolished_regions,
    ]

    need_header = write_header or not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow(row)


def export_batch_json(
    path: str | Path,
    results: list[dict[str, Any]],
) -> None:
    """Write a JSON array of per-pair results."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
