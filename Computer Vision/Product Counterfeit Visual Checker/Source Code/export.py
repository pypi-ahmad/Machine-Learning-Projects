"""Product Counterfeit Visual Checker — result export helpers.

Export screening results to JSON and CSV formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from comparator import ScreeningResult


def export_json(
    result: ScreeningResult,
    output: str | Path,
) -> Path:
    """Write a screening result to a JSON file."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "disclaimer": (
            "This is a visual screening result, not a definitive counterfeit "
            "determination. Scores indicate visual similarity to approved "
            "references. Further investigation is required for any conclusion."
        ),
        "suspect": result.suspect_path,
        "risk_level": result.risk_level,
        "mismatch_risk_pct": result.mismatch_risk_pct,
        "best_composite_score": round(result.best_composite, 6),
        "best_reference": result.best_reference,
        "best_product": result.best_product,
        "comparisons": [
            {
                "reference_path": d.reference_path,
                "reference_product": d.reference_product,
                "global_score": round(d.global_score, 6),
                "region_score": round(d.region_score, 6),
                "histogram_score": round(d.histogram_score, 6),
                "composite_score": round(d.composite_score, 6),
                "region_scores": [round(s, 4) for s in d.region_scores],
            }
            for d in result.details
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def export_csv(
    result: ScreeningResult,
    output: str | Path,
) -> Path:
    """Write a screening result to a CSV file."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "suspect", "risk_level", "reference_path", "reference_product",
            "global_score", "region_score", "histogram_score", "composite_score",
        ])
        for d in result.details:
            writer.writerow([
                result.suspect_path, result.risk_level,
                d.reference_path, d.reference_product,
                round(d.global_score, 6), round(d.region_score, 6),
                round(d.histogram_score, 6), round(d.composite_score, 6),
            ])
    return out


def export_batch_csv(
    results: list[ScreeningResult],
    output: str | Path,
) -> Path:
    """Write batch screening results to CSV."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "suspect", "risk_level", "mismatch_risk_pct",
            "best_composite", "best_reference", "best_product",
        ])
        for r in results:
            writer.writerow([
                r.suspect_path, r.risk_level, r.mismatch_risk_pct,
                round(r.best_composite, 6), r.best_reference, r.best_product,
            ])
    return out
