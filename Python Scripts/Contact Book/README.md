# Contact Book

## Overview

Contact Book is a local Streamlit app for adding, searching, viewing, deleting, and exporting contacts. Contact data is stored in `contacts.json` beside `main.py`.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run streamlit run main.py
```

## Privacy

Contacts can contain personal information. `contacts.json` is ignored by Git but remains readable locally. Use the CSV and JSON download controls only where it is safe to store or share the exported data.

## Verification

Run `uv run python -m py_compile main.py` to check syntax. The Streamlit app can be tested headlessly with `AppTest`; no browser or server is required.
