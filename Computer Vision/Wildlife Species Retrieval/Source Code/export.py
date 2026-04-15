"""Wildlife Species Retrieval -- structured export (JSON / CSV)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from index import SearchHit


def export_json(
    query_path: str | None,
    hits: Sequence[SearchHit],
    output_path: str | Path,
    species_votes: dict[str, float] | None = None,
) -> Path:
    """Export retrieval results to JSON."""
    data = {
        "query": query_path or "",
        "matches": [
            {
                "rank": h.rank,
                "path": h.path,
                "species": h.species,
                "score": round(h.score, 6),
            }
            for h in hits
        ],
    }
    if species_votes:
        data["species_votes"] = {
            k: round(v, 6) for k, v in species_votes.items()
        }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def export_csv(
    query_path: str | None,
    hits: Sequence[SearchHit],
    output_path: str | Path,
) -> Path:
    """Export retrieval results to CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query", "rank", "match_path", "species", "score"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in hits:
            writer.writerow({
                "query": query_path or "",
                "rank": h.rank,
                "match_path": h.path,
                "species": h.species,
                "score": round(h.score, 6),
            })
    return p
