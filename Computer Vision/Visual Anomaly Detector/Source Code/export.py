"""Visual Anomaly Detector — export utilities.
"""Visual Anomaly Detector — export utilities.

Export scored results to JSON and CSV.

Usage::

    from export import export_results

    export_results(results, "output/results.json")
    export_results(results, "output/results.csv", fmt="csv")
"""
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("visual_anomaly.export")

CSV_FIELDS = [
    "source",
    "label",
    "is_anomaly",
    "anomaly_score",
    "mahalanobis",
    "knn",
    "threshold",
]


def export_results(
    results: list[dict],
    path: str | Path,
    *,
    fmt: str = "json",
) -> Path:
    """Export inference results to file.
    """Export inference results to file.

    Parameters
    ----------
    results : list[dict]
        List of prediction outputs.
    path : str | Path
        Output file path.
    fmt : str
        ``"json"`` or ``"csv"``.

    Returns
    -------
    Path
        Path to exported file.
    """
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    else:
        out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    log.info("Exported %d results -> %s", len(results), out)
    return out
