"""Ecommerce Item Attribute Tagger — export utilities.

Export predicted attributes to JSON and CSV files suitable for
catalog enrichment / product information management.

Usage::

    from export import export_catalog, export_csv

    export_catalog(results, "output/catalog.json")
    export_csv(results, "output/catalog.csv")
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("attribute_tagger.export")


def export_catalog(
    results: list[dict],
    path: str | Path,
) -> Path:
    """Export results to structured JSON for catalog enrichment.

    Each result entry has ``source`` and ``attributes`` (structured dict).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    catalog = []
    for r in results:
        entry = {"source": r.get("source", ""),  "attributes": {}}
        pred = r.get("prediction", {})
        for attr_name, info in pred.items():
            entry["attributes"][attr_name] = {
                "value": info.get("label", ""),
                "confidence": info.get("confidence", 0.0),
            }
        catalog.append(entry)

    out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    log.info("Exported %d items → %s", len(catalog), out)
    return out


def export_csv(
    results: list[dict],
    path: str | Path,
) -> Path:
    """Export results to flat CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        log.warning("No results to export")
        return out

    # Gather all attribute names
    attr_names: list[str] = []
    for r in results:
        for k in r.get("prediction", {}):
            if k not in attr_names:
                attr_names.append(k)

    fieldnames = ["source"] + attr_names + [f"{a}_conf" for a in attr_names]

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row: dict[str, str] = {"source": r.get("source", "")}
            pred = r.get("prediction", {})
            for attr in attr_names:
                info = pred.get(attr, {})
                row[attr] = info.get("label", "")
                row[f"{attr}_conf"] = f"{info.get('confidence', 0.0):.4f}"
            writer.writerow(row)

    log.info("Exported %d items → %s (CSV)", len(results), out)
    return out
