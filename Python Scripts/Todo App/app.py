"""Manage persistent todo tasks from the command line."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_DATABASE = Path(__file__).with_name("test.db")


def connect_database(path: Path) -> sqlite3.Connection:
    """Open a task database and create its table for a new installation."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS todo (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            pub_date TEXT NOT NULL
        )
        """
    )
    return connection


def add_task(connection: sqlite3.Connection, content: str) -> int:
    """Create one incomplete task and return its ID."""
    text = content.strip()
    if not text:
        raise ValueError("Task content cannot be empty.")
    cursor = connection.execute(
        "INSERT INTO todo (content, completed, pub_date) VALUES (?, 0, ?)",
        (text, datetime.now(UTC).isoformat()),
    )
    connection.commit()
    return cursor.lastrowid


def task_rows(connection: sqlite3.Connection, include_completed: bool) -> list[sqlite3.Row]:
    """Return open tasks, or every task when requested."""
    where = "" if include_completed else "WHERE completed = 0"
    return connection.execute(f"SELECT id, content, completed, pub_date FROM todo {where} ORDER BY id").fetchall()


def change_task_state(connection: sqlite3.Connection, task_id: int, completed: bool) -> bool:
    """Set a task's completion flag and report whether it existed."""
    cursor = connection.execute("UPDATE todo SET completed = ? WHERE id = ?", (int(completed), task_id))
    connection.commit()
    return cursor.rowcount == 1


def delete_task(connection: sqlite3.Connection, task_id: int) -> bool:
    """Delete one task and report whether it existed."""
    cursor = connection.execute("DELETE FROM todo WHERE id = ?", (task_id,))
    connection.commit()
    return cursor.rowcount == 1


def print_tasks(rows: list[sqlite3.Row]) -> None:
    """Print concise task rows for a terminal session."""
    for row in rows:
        marker = "x" if row["completed"] else " "
        print(f"[{marker}] {row['id']}: {row['content']} ({row['pub_date']})")
    print(f"{len(rows)} task(s).")


def main() -> None:
    """Parse a task command and apply it to the selected SQLite database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="create a task")
    add.add_argument("content")
    listing = commands.add_parser("list", help="list tasks")
    listing.add_argument("--all", action="store_true", help="include completed tasks")
    done = commands.add_parser("done", help="mark a task complete")
    done.add_argument("task_id", type=int)
    reopen = commands.add_parser("reopen", help="mark a task incomplete")
    reopen.add_argument("task_id", type=int)
    delete = commands.add_parser("delete", help="delete a task")
    delete.add_argument("task_id", type=int)
    args = parser.parse_args()

    try:
        with connect_database(args.database) as connection:
            if args.command == "add":
                print(f"Added task {add_task(connection, args.content)}.")
            elif args.command == "list":
                print_tasks(task_rows(connection, args.all))
            elif args.command == "done":
                print("Task completed." if change_task_state(connection, args.task_id, True) else "Task not found.")
            elif args.command == "reopen":
                print("Task reopened." if change_task_state(connection, args.task_id, False) else "Task not found.")
            else:
                print("Task deleted." if delete_task(connection, args.task_id) else "Task not found.")
    except (OSError, sqlite3.Error, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
