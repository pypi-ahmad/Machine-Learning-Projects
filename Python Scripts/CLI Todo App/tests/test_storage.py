"""Unit tests for cli_todo.storage."""

from __future__ import annotations

import json
from pathlib import Path

from cli_todo.models import Task
from cli_todo.storage import TaskStore


class TestTaskStoreEmpty:
    """Tests on a freshly created store with no pre-existing data."""

    def test_list_empty(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        assert store.list_tasks() == []

    def test_add_returns_task(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        t = store.add("Buy milk")
        assert t.id == 0
        assert t.text == "Buy milk"
        assert t.created_at  # non-empty

    def test_add_auto_increments(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        t1 = store.add("First")
        t2 = store.add("Second")
        assert t1.id == 0
        assert t2.id == 1

    def test_add_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        store = TaskStore(path=path)
        store.add("Persisted")

        # Re-open from disk
        store2 = TaskStore(path=path)
        tasks = store2.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].text == "Persisted"

    def test_remove_returns_task(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        t = store.add("To remove")
        removed = store.remove(t.id)
        assert removed is not None
        assert removed.text == "To remove"
        assert store.list_tasks() == []

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        assert store.remove(999) is None

    def test_get_existing(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        t = store.add("Find me")
        assert store.get(t.id) is not None
        assert store.get(t.id).text == "Find me"  # type: ignore[union-attr]

    def test_get_missing(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        assert store.get(42) is None

    def test_clear(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        store.add("A")
        store.add("B")
        count = store.clear()
        assert count == 2
        assert store.list_tasks() == []

    def test_clear_resets_id_counter(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        store.add("A")
        store.clear()
        t = store.add("After clear")
        assert t.id == 0

    def test_list_sorted_by_id(self, tmp_path: Path) -> None:
        store = TaskStore(path=tmp_path / "tasks.json")
        store.add("First")
        store.add("Second")
        store.add("Third")
        ids = [t.id for t in store.list_tasks()]
        assert ids == [0, 1, 2]


class TestTaskStorePersistence:
    """Tests validating file format and corruption handling."""

    def test_json_file_format(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        store = TaskStore(path=path)
        store.add("Check format")

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "next_id" in raw
        assert "tasks" in raw
        assert raw["next_id"] == 1
        assert len(raw["tasks"]) == 1
        assert raw["tasks"][0]["text"] == "Check format"

    def test_corrupt_file_starts_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text("NOT VALID JSON{{{", encoding="utf-8")

        store = TaskStore(path=path)
        assert store.list_tasks() == []

    def test_empty_file_starts_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text("", encoding="utf-8")

        store = TaskStore(path=path)
        assert store.list_tasks() == []


class TestLegacyMigration:
    """Tests for migrating from the old todo.txt format."""

    def test_migrate_empty_legacy(self, tmp_path: Path) -> None:
        legacy = tmp_path / "todo.txt"
        legacy.write_text("0\n", encoding="utf-8")

        json_path = tmp_path / "tasks.json"
        store = TaskStore(path=json_path)
        assert store.list_tasks() == []

    def test_migrate_with_tasks(self, tmp_path: Path) -> None:
        legacy = tmp_path / "todo.txt"
        legacy.write_text("3\n0```Buy milk\n1```Walk dog\n2```Code review\n", encoding="utf-8")

        json_path = tmp_path / "tasks.json"
        store = TaskStore(path=json_path)
        tasks = store.list_tasks()
        assert len(tasks) == 3
        assert tasks[0].text == "Buy milk"
        assert tasks[1].text == "Walk dog"
        assert tasks[2].text == "Code review"

    def test_migrate_preserves_ids(self, tmp_path: Path) -> None:
        legacy = tmp_path / "todo.txt"
        # Simulates: tasks 0 and 2 exist (1 was deleted)
        legacy.write_text("3\n0```First\n2```Third\n", encoding="utf-8")

        json_path = tmp_path / "tasks.json"
        store = TaskStore(path=json_path)
        ids = [t.id for t in store.list_tasks()]
        assert ids == [0, 2]


class TestTaskModel:
    """Tests for Task dataclass serialization."""

    def test_round_trip(self) -> None:
        t = Task(id=5, text="Test task", created_at="2026-01-01T00:00:00+00:00")
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.id == t.id
        assert t2.text == t.text
        assert t2.created_at == t.created_at
