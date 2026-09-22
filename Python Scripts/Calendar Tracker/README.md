# Calendar Tracker

## Overview

Calendar Tracker is a local Streamlit app for creating, browsing, filtering, and deleting personal events. Events are stored in `events.json` next to the application and are not uploaded anywhere.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run streamlit run main.py
```

## Features

- Add titled events with a date, time, category, duration, and optional notes.
- Browse monthly events with previous, next, and today controls.
- Filter all events by category and review events occurring within seven days.
- Require a second confirmation before deleting all local events.

## Data

`events.json` is created in this folder after the first event is saved. It is ignored by Git because it can contain personal calendar details. Deleting that file permanently removes the saved events.

## Verification

Run `uv run python -m py_compile main.py` to check syntax. The app can also be exercised headlessly with Streamlit's `AppTest`; no browser or server is required.
