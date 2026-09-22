# Disk Usage Analyzer

A read-only CLI for inspecting directory sizes, extensions, large files, old files, and available drive space.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python main.py
```

To show top-level subdirectory usage directly:

```powershell
uv run python main.py "C:\path\to\directory"
```

## Interactive options

- Top subdirectories by recursively calculated size.
- File-size totals grouped by extension.
- Largest files in a directory tree.
- Files older than a chosen number of days.
- Available space for local drives or partitions.

## Notes

- The tool only reads metadata and does not modify or delete files.
- Directory-size, extension, large-file, and age reports recursively scan the selected tree. Large or protected locations can take time and may omit unreadable paths.
- Output uses ASCII charts so it displays correctly in standard Windows terminals.

## Project files

```text
main.py         # CLI entry point and analysis logic
pyproject.toml  # uv project definition
uv.lock         # Resolved Python environment
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
