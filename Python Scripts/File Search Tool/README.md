# File Search Tool

A dependency-free, read-only interactive tool for searching an explicitly chosen directory by filename, text content, extension, size, or modification date.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose a search mode, then enter the directory to inspect. A directory path is required; the tool does not default to scanning the current working directory.

## Search behavior

- Name searches accept regular expressions. Invalid expressions are treated as literal text.
- Content searches are literal, case-insensitive by default, and can be limited by extension.
- Content searches skip symlinks and files larger than 2 MB, and stop after 500 matches to keep exploration bounded.
- Recursive searches walk directories without following directory symlinks.

The tool never writes search results, modifies files, or uploads content. Search output can expose sensitive filenames and text snippets, so review it before sharing.
