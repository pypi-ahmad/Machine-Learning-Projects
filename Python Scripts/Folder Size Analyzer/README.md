# Folder Size Analyzer

Read-only command-line tool that reports top-level directory sizes, file extensions, and largest files.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py "C:\path\to\folder" --top 20
```

Omit the path for interactive mode. Use `q` to leave it.

## Behavior

- Requires Python 3.13 or later and no third-party packages.
- Ignores symbolic links and inaccessible entries.
- Reads metadata and file sizes only; it never changes the analyzed folder.
