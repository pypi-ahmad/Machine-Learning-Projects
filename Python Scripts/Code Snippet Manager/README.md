# Code Snippet Manager

## Overview

Code Snippet Manager is an interactive command-line tool for storing, searching, tagging, displaying, copying, editing, and deleting local code snippets. Snippets are saved in `snippets.json` beside `main.py`.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py
uv run python main.py list --lang python
uv run python main.py search "read json"
```

Run `uv run python main.py --help` for the available subcommands.

## Data and privacy

`snippets.json` is ignored by Git. Do not save secrets, API keys, passwords, or private source code unless local storage is appropriate for them. The `copy` command writes the selected code to the system clipboard.

## Verification

Run `uv run python -m py_compile main.py` to check syntax. The help and search/list commands do not require clipboard access.
