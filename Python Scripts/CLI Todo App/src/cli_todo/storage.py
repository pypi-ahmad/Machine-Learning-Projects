"""JSON-backed task storage with legacy migration support."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from platformdirs import user_data_dir

from cli_todo.models import Task

logger = logging.getLogger(__name__)

_APP_NAME = "cli_todo"
_DEFAULT_FILENAME = "tasks.json"


def default_data_path() -> Path:
    """Return the platform-appropriate default path for storing tasks.

    On Windows this is typically:
        ``%LOCALAPPDATA%/cli_todo/tasks.json``
    """
    return Path(user_data_dir(_APP_NAME, appauthor=False, ensure_exists=True)) / _DEFAULT_FILENAME


# ── Persistence data structure ────────────────────────────────────────


class _StoreData:
    """In-memory representation of the JSON file."""

    def __init__(self, next_id: int, tasks: list[Task]) -> None:
        self.next_id = next_id
        self.tasks = tasks

    def to_dict(self) -> dict[str, object]:
        return {
            "next_id": self.next_id,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> _StoreData:
        next_id = int(data.get("next_id", 0))  # type: ignore[arg-type]
        raw_tasks: list[dict[str, object]] = data.get("tasks", [])  # type: ignore[assignment]
        tasks = [Task.from_dict(t) for t in raw_tasks]
        return cls(next_id=next_id, tasks=tasks)

    @classmethod
    def empty(cls) -> _StoreData:
        return cls(next_id=0, tasks=[])


# ── Public API ────────────────────────────────────────────────────────


class TaskStore:
    """Manage tasks persisted as a JSON file.

    Parameters
    ----------
    path:
        Explicit path to the JSON file.  When *None* the platform default
        from :func:`default_data_path` is used.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path: Path = path or default_data_path()
        self._data: _StoreData = self._load()

    # ── read ──────────────────────────────────────────────────────────

    def _load(self) -> _StoreData:
        """Load the store from disk, migrating legacy format if needed."""
        if not self.path.exists():
            # Check for legacy todo.txt next to the JSON path
            legacy = self.path.parent / "todo.txt"
            if legacy.exists():
                return self._migrate_legacy(legacy)
            logger.debug("No existing data at %s - starting fresh.", self.path)
            return _StoreData.empty()

        text = self.path.read_text(encoding="utf-8").strip()
        if not text:
            return _StoreData.empty()

        try:
            raw = json.loads(text)
            data = _StoreData.from_dict(raw)
            logger.debug("Loaded %d tasks from %s", len(data.tasks), self.path)
            return data
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Corrupt data file %s - starting fresh.", self.path)
            return _StoreData.empty()

    # ── write ─────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist current state to the JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.debug("Saved %d tasks to %s", len(self._data.tasks), self.path)

    # ── CRUD ──────────────────────────────────────────────────────────

    def list_tasks(self) -> list[Task]:
        """Return all tasks (ordered by ID)."""
        return sorted(self._data.tasks, key=lambda t: t.id)

    def add(self, text: str) -> Task:
        """Create a new task and persist it.  Returns the created task."""
        task = Task(id=self._data.next_id, text=text)
        self._data.next_id += 1
        self._data.tasks.append(task)
        self._save()
        return task

    def get(self, task_id: int) -> Task | None:
        """Look up a task by ID.  Returns *None* if not found."""
        for t in self._data.tasks:
            if t.id == task_id:
                return t
        return None

    def remove(self, task_id: int) -> Task | None:
        """Remove a task by ID.  Returns the removed task or *None*."""
        for i, t in enumerate(self._data.tasks):
            if t.id == task_id:
                removed = self._data.tasks.pop(i)
                self._save()
                return removed
        return None

    def clear(self) -> int:
        """Remove **all** tasks and reset the ID counter.  Returns count removed."""
        count = len(self._data.tasks)
        self._data = _StoreData.empty()
        self._save()
        return count

    # ── Legacy migration ──────────────────────────────────────────────

    @staticmethod
    def _migrate_legacy(legacy_path: Path) -> _StoreData:
        """Parse the old ``todo.txt`` format and return a new *_StoreData*.

        Legacy format
        -------------
        Line 1 : next-ID counter (int)
        Lines 2+: ``<id>```<task text>``  (triple-backtick delimiter)
        """
        logger.info("Migrating legacy todo.txt at %s", legacy_path)
        lines = legacy_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return _StoreData.empty()

        try:
            next_id = int(lines[0].strip())
        except ValueError:
            next_id = 0

        tasks: list[Task] = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if "```" in line:
                parts = line.split("```", maxsplit=1)
                try:
                    tid = int(parts[0])
                except ValueError:
                    continue
                text = parts[1] if len(parts) > 1 else ""
                tasks.append(Task(id=tid, text=text))

        logger.info("Migrated %d tasks from legacy format.", len(tasks))
        return _StoreData(next_id=next_id, tasks=tasks)
