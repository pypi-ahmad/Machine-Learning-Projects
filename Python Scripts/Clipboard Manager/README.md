# Clipboard Manager

## Overview

Clipboard Manager is a local interactive CLI for manually adding, searching, pinning, copying, exporting, and optionally watching clipboard text. It stores up to 100 recent entries next to `main.py`.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py
```

Use `watch` in the interactive menu to capture clipboard changes. Press Ctrl+C to stop watching.

## Commands

- `list [n]`: display recent entries.
- `add <text>`: store text manually.
- `copy <position>`: copy a stored entry to the system clipboard.
- `del <position>` and `pin <position>`: remove or toggle a stored entry.
- `find <text>`: search stored text.
- `clear`: remove unpinned entries.
- `export`: write the current history to `clipboard_export.json`.

## Privacy

Clipboard entries can contain passwords, access tokens, or private text. The history and export files are ignored by Git, but remain readable on the local machine until manually cleared or deleted. Avoid `watch` on shared or untrusted systems.

## Verification

Run `uv run python -m py_compile main.py` to check syntax. Clipboard access is not needed for the core history helpers.
