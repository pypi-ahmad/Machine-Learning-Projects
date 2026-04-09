"""Data models for cli-todo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Task:
    """A single todo item."""

    id: int
    text: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""
        return {"id": self.id, "text": self.text, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Task:
        """Deserialize from a JSON-friendly dict."""
        return cls(
            id=int(data["id"]),  # type: ignore[arg-type]
            text=str(data["text"]),
            created_at=str(data.get("created_at", "")),
        )
