# Directory Tree Viewer

A dependency-free CLI for displaying a directory tree, optionally including file sizes or filtering files by extension. It can also export a tree report to a text file.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python main.py
```

To print a directory directly, provide its path:

```powershell
uv run python main.py "C:\path\to\directory"
```

## Interactive options

- View a tree with an optional maximum depth and hidden-file setting.
- View a tree with file sizes.
- Filter files by one or more extensions; input is case-insensitive.
- Export the displayed tree and summary to a user-selected text file.

The CLI uses ASCII branch markers so it works reliably in standard Windows terminals. Scanning a large or protected directory can take time or omit unreadable entries.

## Project files

```text
main.py         # CLI entry point and tree-building logic
pyproject.toml  # uv project definition
uv.lock         # Resolved Python environment
```

## Verification

```powershell
uv run python main.py .
uv run python -m py_compile main.py
uv lock --check
```
