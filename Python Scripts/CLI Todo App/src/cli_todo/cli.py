"""CLI interface for cli-todo - built with Typer."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from cli_todo.storage import TaskStore

# ── Logging bootstrap ─────────────────────────────────────────────────

_LOG_FORMAT = "%(levelname)s: %(message)s"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format=_LOG_FORMAT, level=level, stream=sys.stderr)


# ── Typer app ─────────────────────────────────────────────────────────

app = typer.Typer(
    name="cli_todo",
    help="A simple CLI todo app.",
    no_args_is_help=True,
    add_completion=False,
)

# Shared state attached via typer callback
_store: TaskStore | None = None


def _get_store() -> TaskStore:
    """Return the lazily-initialised global store."""
    if _store is None:  # pragma: no cover - only when callback wasn't invoked
        return TaskStore()
    return _store


@app.callback()
def main(
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Path to JSON data file (overrides default)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging to stderr."),
    ] = False,
) -> None:
    """Simple CLI Todo App - add, list, and finish tasks."""
    _configure_logging(verbose)
    global _store
    _store = TaskStore(path=file)


# ── Commands ──────────────────────────────────────────────────────────


@app.command()
def add(
    text: Annotated[
        str | None,
        typer.Argument(help="Task description.  Omit to be prompted."),
    ] = None,
) -> None:
    """Add a new task."""
    store = _get_store()
    if text is None:
        text = typer.prompt("Enter task to add")
    task = store.add(text)
    typer.echo(f'Added task "{task.text}" with ID {task.id}')


@app.command(name="list")
def list_tasks(
    as_json: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON."),
    ] = False,
) -> None:
    """Display all tasks."""
    store = _get_store()
    tasks = store.list_tasks()

    if as_json:
        typer.echo(json.dumps([t.to_dict() for t in tasks], indent=2, ensure_ascii=False))
        return

    if not tasks:
        typer.echo("No tasks yet! Use `add` to create one.")
        return

    typer.echo("YOUR TASKS\n**********")
    for t in tasks:
        typer.echo(f"• {t.text} (ID: {t.id})")
    typer.echo("")


@app.command()
def done(
    task_id: Annotated[
        int | None,
        typer.Argument(help="ID of the task to finish.  Omit to be prompted."),
    ] = None,
) -> None:
    """Mark a task as finished (removes it)."""
    store = _get_store()
    if task_id is None:
        task_id = typer.prompt("Enter ID of task to finish", type=int)
    removed = store.remove(task_id)
    if removed:
        typer.echo(f'Finished and removed task "{removed.text}" with ID {removed.id}')
    else:
        typer.echo(f"Error: no task with ID {task_id}", err=True)
        raise typer.Exit(code=1)


@app.command()
def clear(
    force: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """Remove all tasks."""
    store = _get_store()
    if not force:
        typer.confirm("Remove ALL tasks?", abort=True)
    count = store.clear()
    typer.echo(f"Cleared {count} task(s).")
