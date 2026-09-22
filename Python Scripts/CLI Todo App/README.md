# cli-todo

A simple, modern CLI todo app.  Add, list, finish, and clear tasks from your
terminal.  Tasks are stored as JSON in a platform-appropriate data directory.

## Requirements

- Python ≥ 3.11

## Installation

```powershell
uv sync
```

Run commands with `uv run`; no environment activation is needed.

## Usage

```bash
# Add a task (inline)
uv run cli_todo add "Buy groceries"

# Add a task (prompted)
uv run cli_todo add

# List all tasks
uv run cli_todo list

# List as JSON (machine-readable)
uv run cli_todo list --json

# Finish / remove a task by ID
uv run cli_todo done 0

# Finish a task (prompted for ID)
uv run cli_todo done

# Remove all tasks (with confirmation prompt)
uv run cli_todo clear

# Remove all tasks (skip prompt)
uv run cli_todo clear --yes

# Show help
uv run cli_todo --help
uv run cli_todo add --help
```

### Global options

| Option | Description |
|--------|-------------|
| `--file PATH` / `-f PATH` | Override the JSON data file location. |
| `--verbose` / `-v` | Enable debug logging to stderr. |
| `--help` | Show help and exit. |

### Commands

| Command | Description |
|---------|-------------|
| `add [TEXT]` | Add a new task. Prompts if TEXT is omitted. |
| `list` | Display all tasks. Use `--json` for JSON output. |
| `done [ID]` | Mark a task as finished (removes it). Prompts if ID is omitted. |
| `clear` | Remove **all** tasks. Use `--yes` to skip confirmation. |

## Data storage

Tasks are persisted as a JSON file.  The default location is determined by
[platformdirs](https://pypi.org/project/platformdirs/):

| OS | Default path |
|----|-------------|
| Windows | `%LOCALAPPDATA%\cli_todo\tasks.json` |
| macOS | `~/Library/Application Support/cli_todo/tasks.json` |
| Linux | `~/.local/share/cli_todo/tasks.json` |

Override with `--file`:

```bash
uv run cli_todo --file ./my-tasks.json list
```

### Legacy migration

If you previously used the old `todo.txt` format, place the file next to
the JSON path and `cli-todo` will auto-migrate it on first run.

## Development

```powershell
uv sync
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/
```

## License

MIT
