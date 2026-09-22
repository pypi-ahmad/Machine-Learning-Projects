# Recent Files Tracker

List recently modified files in a selected directory.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py . --days 7 --ext .py .md --top 20
```

Without a path, the tool prompts for a directory and optionally lets you open a selected result. Scanning reads file metadata; opening a file launches the system-associated application. Use only paths you intend to inspect.
