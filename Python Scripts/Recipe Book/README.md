# Recipe Book

A local Tkinter desktop app for storing, finding, editing, and deleting recipes.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Recipes are stored in `recipes.json` beside `main.py` after the first save. The app runs locally and makes no network requests; treat the recipe file as personal data and do not commit it.
