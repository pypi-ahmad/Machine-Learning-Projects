"""CLI integration tests using typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cli_todo.cli import app

runner = CliRunner()


def _invoke(*args: str, data_file: Path | None = None):
    """Helper: invoke the CLI, optionally pointing --file at a temp path."""
    cli_args: list[str] = []
    if data_file is not None:
        cli_args += ["--file", str(data_file)]
    cli_args += list(args)
    return runner.invoke(app, cli_args)


class TestAdd:
    def test_add_with_argument(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        result = _invoke("add", "Buy groceries", data_file=f)
        assert result.exit_code == 0
        assert 'Added task "Buy groceries" with ID 0' in result.output

    def test_add_prompts_when_no_arg(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        result = runner.invoke(app, ["--file", str(f), "add"], input="From prompt\n")
        assert result.exit_code == 0
        assert "From prompt" in result.output

    def test_add_multiple(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "First", data_file=f)
        result = _invoke("add", "Second", data_file=f)
        assert "ID 1" in result.output


class TestList:
    def test_list_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        result = _invoke("list", data_file=f)
        assert result.exit_code == 0
        assert "No tasks yet" in result.output

    def test_list_with_tasks(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "Alpha", data_file=f)
        _invoke("add", "Beta", data_file=f)
        result = _invoke("list", data_file=f)
        assert "Alpha" in result.output
        assert "Beta" in result.output
        assert "YOUR TASKS" in result.output

    def test_list_json(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "JSON task", data_file=f)
        result = _invoke("list", "--json", data_file=f)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["text"] == "JSON task"

    def test_list_json_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        result = _invoke("list", "--json", data_file=f)
        assert result.exit_code == 0
        assert json.loads(result.output) == []


class TestDone:
    def test_done_existing(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "To finish", data_file=f)
        result = _invoke("done", "0", data_file=f)
        assert result.exit_code == 0
        assert "Finished and removed" in result.output
        assert "To finish" in result.output

    def test_done_nonexistent(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        result = _invoke("done", "999", data_file=f)
        assert result.exit_code == 1
        assert "Error: no task with ID 999" in result.output

    def test_done_removes_from_list(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "Gone soon", data_file=f)
        _invoke("done", "0", data_file=f)
        result = _invoke("list", data_file=f)
        assert "Gone soon" not in result.output

    def test_done_prompts_when_no_arg(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "Prompt done", data_file=f)
        result = runner.invoke(app, ["--file", str(f), "done"], input="0\n")
        assert result.exit_code == 0
        assert "Finished and removed" in result.output


class TestClear:
    def test_clear_with_confirm(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "A", data_file=f)
        _invoke("add", "B", data_file=f)
        result = runner.invoke(app, ["--file", str(f), "clear"], input="y\n")
        assert result.exit_code == 0
        assert "Cleared 2 task(s)" in result.output

    def test_clear_with_yes_flag(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "A", data_file=f)
        result = _invoke("clear", "--yes", data_file=f)
        assert result.exit_code == 0
        assert "Cleared 1 task(s)" in result.output

    def test_clear_abort(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        _invoke("add", "Keep me", data_file=f)
        result = runner.invoke(app, ["--file", str(f), "clear"], input="n\n")
        assert result.exit_code != 0  # typer.Abort raises SystemExit(1)


class TestGlobalOptions:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Typer's rich formatting may alter whitespace; just check key words
        assert "todo" in result.output.lower()
        assert "add" in result.output.lower()
        assert "list" in result.output.lower()
        assert "done" in result.output.lower()

    def test_verbose(self, tmp_path: Path) -> None:
        f = tmp_path / "tasks.json"
        result = _invoke("--verbose", "add", "Debug test", data_file=f)
        assert result.exit_code == 0
