# Meme Fetcher

A standard-library CLI that retrieves public meme metadata from Meme API and can save local favourites.

```powershell
uv sync --no-config
uv run --no-config python main.py --count 3
```

Fetching memes contacts Meme API. Favourites are stored locally in `meme_favorites.json` beside `main.py`; no API request was made during local verification.
