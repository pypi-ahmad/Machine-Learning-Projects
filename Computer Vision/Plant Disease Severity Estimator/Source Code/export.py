"""Plant Disease Severity Estimator -- structured export (JSON / CSV)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from classifier import PredictionResult


def _result_row(r: PredictionResult, source: str = "") -> dict:
    """Flatten a PredictionResult into a serialisable dict."""
    return {
        "source": source,
        "class_name": r.class_name,
        "plant": r.plant,
        "disease": r.disease,
        "severity_index": r.severity_index,
        "severity_name": r.severity_name,
        "confidence": round(r.confidence, 4),
        "lesion_ratio": round(r.lesion_ratio, 4) if r.lesion_ratio is not None else None,
    }


def export_json(
    results: Sequence[PredictionResult],
    output_path: str | Path,
    sources: Sequence[str] | None = None,
) -> Path:
    """Export results to a JSON file."""
    names = sources or [""] * len(results)
    rows = [_result_row(r, s) for r, s in zip(results, names)]
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return p


def export_csv(
    results: Sequence[PredictionResult],
    output_path: str | Path,
    sources: Sequence[str] | None = None,
) -> Path:
    """Export results to a CSV file."""
    names = sources or [""] * len(results)
    rows = [_result_row(r, s) for r, s in zip(results, names)]
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p
