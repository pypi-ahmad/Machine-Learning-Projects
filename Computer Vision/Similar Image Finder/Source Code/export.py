"""Similar Image Finder — result export helpers.

Export retrieval results to JSON and CSV formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from index import SearchHit


def export_json(
    query_path: str,
    hits: list[SearchHit],
    output: str | Path,
    *,
    category_votes: dict[str, float] | None = None,
) -> Path:
    """Write retrieval results to a JSON file."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": str(query_path),
        "matches": [
            {
                "rank": h.rank,
                "path": h.path,
                "category": h.category,
                "score": round(h.score, 6),
            }
            for h in hits
        ],
    }
    if category_votes:
        payload["category_votes"] = {
            k: round(v, 4)
            for k, v in sorted(category_votes.items(), key=lambda x: -x[1])
        }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def export_csv(
    query_path: str,
    hits: list[SearchHit],
    output: str | Path,
) -> Path:
    """Write retrieval results to a CSV file."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "rank", "match_path", "category", "score"])
        for h in hits:
            writer.writerow([str(query_path), h.rank, h.path, h.category, round(h.score, 6)])
    return out


def export_batch_csv(
    results: list[dict],
    output: str | Path,
) -> Path:
    """Write batch evaluation results to CSV.

    Each dict should have: query_path, hits (list[SearchHit]).
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "rank", "match_path", "category", "score"])
        for r in results:
            q = str(r["query_path"])
            for h in r["hits"]:
                writer.writerow([q, h.rank, h.path, h.category, round(h.score, 6)])
    return out
