# File Size Checker

A dependency-free, read-only interactive tool for reporting file sizes, directory totals, large files, and extension breakdowns.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Enter the exact file or directory path to inspect. The tool supports:

- One file or directory total.
- A direct or recursive size-ranked directory listing.
- Recursive large-file lookup using a threshold such as `500KB` or `1MB`.
- A recursive extension-size breakdown.

Traversal skips symlinks and inaccessible files where possible. The tool does not use filename glob patterns, modify files, create reports, or upload metadata.
