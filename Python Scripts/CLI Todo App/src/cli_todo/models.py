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
        raw_id = data.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
            raise ValueError("Task ID must be an integer.")
        raw_text = data.get("text")
        if not isinstance(raw_text, str):
            raise ValueError("Task text must be a string.")
        raw_created_at = data.get("created_at", "")
        return cls(
            id=int(raw_id),
            text=raw_text,
            created_at=raw_created_at if isinstance(raw_created_at, str) else "",
        )
