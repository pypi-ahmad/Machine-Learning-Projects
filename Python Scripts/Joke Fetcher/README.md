# Joke Fetcher

An interactive CLI that fetches a joke from JokeAPI and falls back to built-in jokes when the API is unavailable.

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Saved favourites are stored locally in `jokes_favourites.json` beside `main.py`. Fetching a live joke contacts JokeAPI; that network path was not run during local verification.
