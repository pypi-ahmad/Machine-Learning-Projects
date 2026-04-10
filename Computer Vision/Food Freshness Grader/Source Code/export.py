"""Food Freshness Grader — result export helpers.

Export grading results to JSON and CSV formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from grader import GradeResult


def export_json(
    results: list[dict],
    output: str | Path,
) -> Path:
    """Write grading results to a JSON file.

    Each dict: {path, freshness, produce, confidence, class_name}.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "total": len(results),
        "fresh_count": sum(1 for r in results if r["freshness"] == "fresh"),
        "stale_count": sum(1 for r in results if r["freshness"] == "stale"),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def export_csv(
    results: list[dict],
    output: str | Path,
) -> Path:
    """Write grading results to a CSV file."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "freshness", "produce", "class_name", "confidence"])
        for r in results:
            writer.writerow([
                r.get("path", ""),
                r["freshness"],
                r["produce"],
                r["class_name"],
                round(r["confidence"], 6),
            ])
    return out


def result_to_dict(
    grade: GradeResult,
    image_path: str | None = None,
) -> dict:
    """Convert a GradeResult to an export-ready dict."""
    return {
        "path": image_path or "",
        "freshness": grade.freshness,
        "produce": grade.produce,
        "class_name": grade.class_name,
        "confidence": round(grade.confidence, 6),
    }
