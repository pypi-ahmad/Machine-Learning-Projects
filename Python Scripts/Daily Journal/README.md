# Daily Journal

A local Streamlit journal for writing one entry per day, adding a mood and tags, browsing saved entries, and exporting them as CSV.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run streamlit run main.py
```

## Data and privacy

- Entries are saved in `journal.json` beside `main.py`, so the storage location does not depend on the directory from which you start the app.
- The journal does not send entries to a remote service.
- `journal.json` and exported CSV files are ignored by Git because they may contain private writing.
- Keep your device and local backup location secure. This app does not encrypt journal entries.

## Features

- Create or replace the entry for a selected day.
- Assign a mood and comma-separated tags.
- Search entries and filter them by tag or mood.
- View entry counts, mood distribution, monthly activity, and a CSV export.

## Project files

```text
main.py         # Streamlit application
pyproject.toml  # uv dependency definition
uv.lock         # Resolved dependency versions
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
