"""Alert management for Driver Drowsiness Monitor.

Consolidates drowsiness signals (prolonged closure, PERCLOS,
yawns, distraction) into alert events with cooldown dedup
and timestamped logging.
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
from config import DrowsinessConfig

log = logging.getLogger("drowsiness.alert_manager")


@dataclass
class AlertEvent:
    """Single alert record."""

    timestamp: str
    alert_type: str         # "prolonged_closure" | "perclos" | "yawn" | "distraction"
    severity: str           # "warning" | "critical"
    detail: str
    session: str


class AlertManager:
    """Cooldown-aware alert manager with logging."""

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self.cfg = cfg
        self._events: list[AlertEvent] = []
        self._last_alert_time: dict[str, float] = {}
        self._session = cfg.session_name or datetime.now(
            timezone.utc,
        ).strftime("session_%Y%m%d_%H%M%S")
        self._active_alerts: set[str] = set()

    @property
    def session_name(self) -> str:
        return self._session

    @property
    def events(self) -> list[AlertEvent]:
        return list(self._events)

    @property
    def count(self) -> int:
        return len(self._events)

    @property
    def active_alerts(self) -> set[str]:
        """Currently active alert types."""
        return set(self._active_alerts)

    def check_and_alert(
        self,
        *,
        prolonged_closure: bool = False,
        drowsy_by_perclos: bool = False,
        perclos: float = 0.0,
        yawn_detected: bool = False,
        distracted: bool = False,
        yaw: float = 0.0,
        ear: float = 0.0,
    ) -> list[AlertEvent]:
        """Evaluate all signals and emit alerts with cooldown.

        Returns
        -------
        list[AlertEvent]
            Newly emitted alerts (if any).
        """
        new_alerts: list[AlertEvent] = []
        self._active_alerts.clear()

        if prolonged_closure:
            self._active_alerts.add("prolonged_closure")
            evt = self._maybe_emit(
                "prolonged_closure",
                "critical",
                f"Eyes closed for extended period (EAR={ear:.2f})",
            )
            if evt:
                new_alerts.append(evt)

        if drowsy_by_perclos:
            self._active_alerts.add("perclos")
            evt = self._maybe_emit(
                "perclos",
                "critical",
                f"PERCLOS={perclos:.1%} exceeds threshold",
            )
            if evt:
                new_alerts.append(evt)

        if yawn_detected:
            self._active_alerts.add("yawn")
            evt = self._maybe_emit(
                "yawn",
                "warning",
                "Yawn detected — possible fatigue",
            )
            if evt:
                new_alerts.append(evt)

        if distracted:
            self._active_alerts.add("distraction")
            evt = self._maybe_emit(
                "distraction",
                "warning",
                f"Looking away from road (yaw={yaw:.1f}°)",
            )
            if evt:
                new_alerts.append(evt)

        return new_alerts

    def _maybe_emit(
        self, alert_type: str, severity: str, detail: str,
    ) -> AlertEvent | None:
        """Emit an alert if cooldown has elapsed."""
        now = time.monotonic()
        last = self._last_alert_time.get(alert_type, 0.0)
        if (now - last) < self.cfg.alert_cooldown_sec:
            return None

        self._last_alert_time[alert_type] = now
        evt = AlertEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            alert_type=alert_type,
            severity=severity,
            detail=detail,
            session=self._session,
        )
        self._events.append(evt)
        log.warning("[%s] %s: %s", severity.upper(), alert_type, detail)
        return evt

    # ── persistence ───────────────────────────────────────

    def save_csv(self, path: str | Path | None = None) -> Path:
        """Export alert log to CSV."""
        out = (
            Path(path) if path
            else Path(self.cfg.log_dir) / f"{self._session}_alerts.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp", "alert_type", "severity", "detail", "session",
                ],
            )
            writer.writeheader()
            for evt in self._events:
                writer.writerow({
                    "timestamp": evt.timestamp,
                    "alert_type": evt.alert_type,
                    "severity": evt.severity,
                    "detail": evt.detail,
                    "session": evt.session,
                })

        log.info("Alert CSV → %s (%d events)", out, len(self._events))
        return out

    def save_json(self, path: str | Path | None = None) -> Path:
        """Export alert log to JSON."""
        out = (
            Path(path) if path
            else Path(self.cfg.log_dir) / f"{self._session}_alerts.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session": self._session,
            "total_alerts": len(self._events),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "events": [
                {
                    "timestamp": e.timestamp,
                    "alert_type": e.alert_type,
                    "severity": e.severity,
                    "detail": e.detail,
                }
                for e in self._events
            ],
        }
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Alert JSON → %s (%d events)", out, len(self._events))
        return out

    def reset(self) -> None:
        self._events.clear()
        self._last_alert_time.clear()
        self._active_alerts.clear()
