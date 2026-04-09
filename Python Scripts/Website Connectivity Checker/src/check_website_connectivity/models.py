"""Data models for website connectivity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class Status(Enum):
    """Result status of a connectivity check."""

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of checking a single URL."""

    url: str
    status: Status
    status_code: int | None = None
    reason: str = ""
    response_time_ms: float | None = None
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "url": self.url,
            "status": self.status.value,
            "status_code": self.status_code,
            "reason": self.reason,
            "response_time_ms": self.response_time_ms,
            "checked_at": self.checked_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | int | float | None]) -> CheckResult:
        return cls(
            url=str(data["url"]),
            status=Status(data["status"]),
            status_code=int(data["status_code"]) if data.get("status_code") is not None else None,
            reason=str(data.get("reason", "")),
            response_time_ms=(
                float(data["response_time_ms"])
                if data.get("response_time_ms") is not None
                else None
            ),
            checked_at=str(data.get("checked_at", "")),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        return self.status is Status.REACHABLE

    def status_label(self) -> str:
        """Human-readable one-word label (backwards-compatible with legacy)."""
        if self.status is Status.REACHABLE:
            return "working"
        return "not working"
