# Todo App

Single-file command-line todo manager backed by local SQLite storage.

## Setup

Requirements: Python 3.13+. This project uses only the standard library.

```powershell
cd "Todo App"
uv sync
```

## Usage

Add a task:

```powershell
uv run python app.py add "Finish project notes"
```

List open tasks:

```powershell
uv run python app.py list
```

Mark a task done, reopen it, or delete it:

```powershell
uv run python app.py done 1
uv run python app.py reopen 1
uv run python app.py delete 1
```

Use `list --all` to include completed tasks. The default database is `test.db` beside `app.py`; pass `--database PATH` before the command to use another SQLite file.

## Behavior

The app stores task content, completion state, and creation timestamp. Commands use parameterized SQLite queries. The existing `test.db` task schema remains compatible.

The old Flask UI has been retired to keep the project within its single-file, local-tool scope. The app does not start a web server or expose tasks over a network.

## Project files

```text
Todo App/
├── app.py
├── test.db
├── pyproject.toml
└── uv.lock
```
