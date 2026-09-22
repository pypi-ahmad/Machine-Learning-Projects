# Recipe Manager

A local Streamlit app for browsing recipes, scaling ingredients, and building shopping lists.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config streamlit run main.py
```

The app starts with in-memory sample recipes. It creates `recipes.json` beside `main.py` only after you save or update data. It runs locally and makes no network requests.
