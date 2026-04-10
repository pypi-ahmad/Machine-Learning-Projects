"""Video Event Search — export utilities.

Export events to JSON and CSV files.  Thin wrapper around
:class:`event_store.EventStore` for convenience usage from the CLI.

Usage::

    from export import export_events

    export_events("outputs/events.json", "outputs/events.csv")
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("video_event_search.export")


def export_events(
    json_path: str | Path,
    csv_path: str | Path | None = None,
    *,
    fmt: str = "json",
) -> Path:
    """Re-export events from a JSON store to another format.

    Parameters
    ----------
    json_path : str | Path
        Path to the source events JSON.
    csv_path : str | Path | None
        Output CSV path (default: same stem as JSON with .csv).
    fmt : str
        Export format — ``"json"`` (pretty-print) or ``"csv"``.

    Returns
    -------
    Path
        Path to the exported file.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Event store not found: {json_path}")

    events = json.loads(json_path.read_text(encoding="utf-8"))
    log.info("Loaded %d events from %s", len(events), json_path)

    if fmt == "csv":
        out = Path(csv_path) if csv_path else json_path.with_suffix(".csv")
        _write_csv(events, out)
        return out

    # Default: pretty-print JSON
    out = Path(csv_path) if csv_path else json_path
    out.write_text(json.dumps(events, indent=2), encoding="utf-8")
    log.info("Exported %d events → %s", len(events), out)
    return out


def _write_csv(events: list[dict], path: Path) -> None:
    if not events:
        log.warning("No events to export")
        return

    fieldnames = list(events[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
    log.info("Exported %d events → %s (CSV)", len(events), path)
