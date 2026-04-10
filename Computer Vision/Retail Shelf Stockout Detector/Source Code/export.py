"""Event and data export — CSV, JSON, and alert snapshots.

Usage::

    from export import EventExporter

    exporter = EventExporter(output_dir="outputs")
    exporter.log_frame(frame, frame_result)
    exporter.flush()
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from zones import AlertEvent, FrameResult


class EventExporter:
    """Accumulates events and exports to CSV, JSON, and snapshot images.

    Parameters
    ----------
    output_dir : str | Path
        Base directory for all exports.
    save_csv : bool
        Write cumulative CSV on :meth:`flush`.
    save_json : bool
        Write cumulative JSON on :meth:`flush`.
    save_snapshots : bool
        Save annotated alert frames as images.
    snapshot_cooldown : float
        Minimum seconds between snapshots for the same zone.
    """

    _CSV_FIELDS = ["timestamp", "zone", "count", "threshold", "event_type"]

    def __init__(
        self,
        output_dir: str | Path = "outputs",
        *,
        save_csv: bool = True,
        save_json: bool = True,
        save_snapshots: bool = True,
        snapshot_cooldown: float = 10.0,
    ) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "snapshots").mkdir(exist_ok=True)

        self._save_csv = save_csv
        self._save_json = save_json
        self._save_snapshots = save_snapshots
        self._cooldown = snapshot_cooldown

        self._events: list[dict] = []
        self._last_snapshot: dict[str, float] = {}  # zone_name -> timestamp

    @property
    def event_count(self) -> int:
        return len(self._events)

    def log_frame(
        self,
        frame: np.ndarray | None,
        result: FrameResult,
        annotated: np.ndarray | None = None,
    ) -> None:
        """Process a frame result, accumulating events and saving snapshots.

        Parameters
        ----------
        frame : np.ndarray | None
            Original frame (used for snapshots).
        result : FrameResult
            Output from :meth:`ZoneCounter.update`.
        annotated : np.ndarray | None
            Annotated frame for snapshot. Falls back to *frame* if None.
        """
        now = time.time()

        for zs in result.zone_statuses:
            if not zs.is_low_stock:
                continue

            event = {
                "timestamp": result.timestamp,
                "zone": zs.name,
                "count": zs.count,
                "threshold": zs.threshold,
                "event_type": "low_stock",
            }
            self._events.append(event)

            # Snapshot with cooldown
            if self._save_snapshots and frame is not None:
                last = self._last_snapshot.get(zs.name, 0)
                if now - last >= self._cooldown:
                    snap = annotated if annotated is not None else frame
                    self._save_snapshot(snap, zs.name, result.timestamp)
                    self._last_snapshot[zs.name] = now

    def flush(self) -> dict[str, str]:
        """Write accumulated data to disk.

        Returns
        -------
        dict
            Paths to written files.
        """
        written: dict[str, str] = {}

        if self._save_csv and self._events:
            csv_path = self._dir / "events.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                writer.writeheader()
                writer.writerows(self._events)
            written["csv"] = str(csv_path)

        if self._save_json and self._events:
            json_path = self._dir / "events.json"
            json_path.write_text(
                json.dumps(self._events, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written["json"] = str(json_path)

        # Summary
        summary_path = self._dir / "summary.json"
        summary = {
            "total_events": len(self._events),
            "zones_triggered": list({e["zone"] for e in self._events}),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        written["summary"] = str(summary_path)

        return written

    def _save_snapshot(self, frame: np.ndarray, zone: str, timestamp: str) -> None:
        """Save an alert snapshot image."""
        safe_zone = zone.replace(" ", "_").replace("/", "_")
        safe_ts = timestamp.replace(":", "-")
        filename = f"alert_{safe_zone}_{safe_ts}.jpg"
        path = self._dir / "snapshots" / filename
        cv2.imwrite(str(path), frame)
