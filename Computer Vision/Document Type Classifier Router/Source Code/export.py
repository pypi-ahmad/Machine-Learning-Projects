"""Document Type Classifier Router — structured export (JSON / CSV)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from classifier import ClassificationResult
from router import RoutingDecision


def _result_row(
    cr: ClassificationResult,
    rd: RoutingDecision,
    source: str = "",
) -> dict:
    return {
        "source": source,
        "document_type": cr.class_name,
        "display_label": cr.display_label,
        "confidence": round(cr.confidence, 4),
        "pipeline": rd.pipeline,
        "routed": rd.routed,
        "reason": rd.reason,
    }


def export_json(
    cls_results: Sequence[ClassificationResult],
    routes: Sequence[RoutingDecision],
    output_path: str | Path,
    sources: Sequence[str] | None = None,
) -> Path:
    names = sources or [""] * len(cls_results)
    rows = [_result_row(c, r, s)
            for c, r, s in zip(cls_results, routes, names)]
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return p


def export_csv(
    cls_results: Sequence[ClassificationResult],
    routes: Sequence[RoutingDecision],
    output_path: str | Path,
    sources: Sequence[str] | None = None,
) -> Path:
    names = sources or [""] * len(cls_results)
    rows = [_result_row(c, r, s)
            for c, r, s in zip(cls_results, routes, names)]
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p
