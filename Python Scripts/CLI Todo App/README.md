# cli-todo

A simple, modern CLI todo app.  Add, list, finish, and clear tasks from your
terminal.  Tasks are stored as JSON in a platform-appropriate data directory.

## Requirements

- Python ≥ 3.11

## Installation

```bash
# From the Cli_todo/ directory:
pip install -e .
```

This installs a `cli_todo` console command.

## Usage

```bash
# Add a task (inline)
cli_todo add "Buy groceries"

# Add a task (prompted)
cli_todo add

# List all tasks
cli_todo list

# List as JSON (machine-readable)
cli_todo list --json

# Finish / remove a task by ID
cli_todo done 0

# Finish a task (prompted for ID)
cli_todo done

# Remove all tasks (with confirmation prompt)
cli_todo clear

# Remove all tasks (skip prompt)
cli_todo clear --yes

# Show help
cli_todo --help
cli_todo add --help
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
cli_todo --file ./my-tasks.json list
```

### Legacy migration

If you previously used the old `todo.txt` format, place the file next to
the JSON path and `cli-todo` will auto-migrate it on first run.

## Development

```bash
pip install -e ".[dev]"   # or: pip install -e . && pip install pytest ruff
ruff check src/ tests/
ruff format src/ tests/
pytest -q
```

## License

MIT
