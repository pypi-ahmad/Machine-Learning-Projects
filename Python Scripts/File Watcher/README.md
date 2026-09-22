# File Watcher

A dependency-free interactive tool for monitoring an explicitly chosen local directory with periodic polling.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose whether to start a watcher, save an in-memory snapshot, or compare two snapshots. Enter the directory path explicitly; the tool does not default to the current directory.

## Watch behavior

- Reports created, modified, and deleted regular files.
- Can include nested directories without following symlinks.
- Requires a polling interval greater than zero.
- Stops with Ctrl+C or an optional maximum event count.
- Optional logs must be outside the watched directory to prevent the log itself from generating repeated events.

The watcher reads metadata only. It does not modify the watched directory, but filenames and paths in console or log output can be sensitive.
