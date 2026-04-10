"""Attendance logging module for Face Verification Attendance System.

Records timestamped attendance entries with session-level dedup
(same person won't be re-logged within a configurable cooldown window).
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceAttendanceConfig

log = logging.getLogger("face_attendance.attendance_log")


@dataclass
class AttendanceEntry:
    """Single attendance record."""

    identity: str
    timestamp: str
    similarity: float
    session: str


class AttendanceLogger:
    """Session-aware attendance logger with dedup cooldown."""

    def __init__(self, cfg: FaceAttendanceConfig) -> None:
        self.cfg = cfg
        self._entries: list[AttendanceEntry] = []
        self._last_logged: dict[str, float] = {}   # identity → epoch
        self._session = cfg.session_name or datetime.now(
            timezone.utc,
        ).strftime("session_%Y%m%d_%H%M%S")

    @property
    def session_name(self) -> str:
        return self._session

    @property
    def entries(self) -> list[AttendanceEntry]:
        return list(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)

    def log(self, identity: str, similarity: float) -> bool:
        """Record attendance for an identity.

        Applies dedup cooldown — if the same person was logged less
        than ``dedup_cooldown_sec`` ago, the entry is skipped.

        Parameters
        ----------
        identity : str
            Matched identity name.
        similarity : float
            Cosine similarity score.

        Returns
        -------
        bool
            True if a new entry was recorded (not deduped).
        """
        if identity == self.cfg.unknown_label:
            return False

        now = time.time()
        last = self._last_logged.get(identity, 0.0)
        if (now - last) < self.cfg.dedup_cooldown_sec:
            return False

        entry = AttendanceEntry(
            identity=identity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            similarity=round(similarity, 4),
            session=self._session,
        )
        self._entries.append(entry)
        self._last_logged[identity] = now

        log.info(
            "Attendance: %s (sim=%.3f) @ %s",
            identity, similarity, entry.timestamp,
        )
        return True

    def recent_identities(self, n: int = 10) -> list[str]:
        """Return last *n* unique logged identities (most recent first)."""
        seen = []
        for entry in reversed(self._entries):
            if entry.identity not in seen:
                seen.append(entry.identity)
            if len(seen) >= n:
                break
        return seen

    def save_csv(self, path: str | Path | None = None) -> Path:
        """Export attendance log to CSV."""
        out = Path(path) if path else Path(self.cfg.log_dir) / f"{self._session}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "identity", "similarity", "session"],
            )
            writer.writeheader()
            for entry in self._entries:
                writer.writerow({
                    "timestamp": entry.timestamp,
                    "identity": entry.identity,
                    "similarity": entry.similarity,
                    "session": entry.session,
                })

        log.info("Attendance CSV → %s (%d entries)", out, len(self._entries))
        return out

    def save_json(self, path: str | Path | None = None) -> Path:
        """Export attendance log to JSON."""
        out = Path(path) if path else Path(self.cfg.log_dir) / f"{self._session}.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session": self._session,
            "total_entries": len(self._entries),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "identity": e.identity,
                    "similarity": e.similarity,
                }
                for e in self._entries
            ],
        }
        out.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )
        log.info("Attendance JSON → %s (%d entries)", out, len(self._entries))
        return out

    def reset(self) -> None:
        """Clear all entries and cooldown state."""
        self._entries.clear()
        self._last_logged.clear()
