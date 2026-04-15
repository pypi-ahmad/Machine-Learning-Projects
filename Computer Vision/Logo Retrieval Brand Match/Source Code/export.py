"""Logo Retrieval Brand Match -- export utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from index import SearchHit


def export_results_json(
    path: str | Path,
    query_path: str | None,
    hits: list[SearchHit],
    brand_votes: dict[str, float] | None = None,
) -> Path:
    """Save retrieval results to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "query": query_path,
        "matches": [
            {
                "rank": h.rank,
                "brand": h.brand,
                "score": round(h.score, 6),
                "path": h.path,
            }
            for h in hits
        ],
    }
    if brand_votes:
        data["brand_votes"] = {k: round(v, 4) for k, v in brand_votes.items()}

    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


class CSVExporter:
    """Append retrieval results row-by-row to a CSV file."""

    _FIELDS = [
        "query", "rank", "brand", "score", "match_path",
    ]

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=self._FIELDS)
        self._writer.writeheader()

    def write_hits(self, query: str, hits: list[SearchHit]) -> None:
        for h in hits:
            self._writer.writerow({
                "query": query,
                "rank": h.rank,
                "brand": h.brand,
                "score": round(h.score, 6),
                "match_path": h.path,
            })

    def close(self) -> None:
        self._fp.close()
