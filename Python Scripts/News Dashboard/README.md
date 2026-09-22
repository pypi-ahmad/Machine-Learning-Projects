# News Dashboard

A local Streamlit dashboard for viewing headlines from RSS feeds or NewsAPI and saving article links for later.

## Requirements

- Python 3.14 or later
- Internet access when loading headlines
- An optional NewsAPI key for NewsAPI mode

## Run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

Without a key, select an RSS feed in the sidebar. To use NewsAPI, set `NEWS_API_KEY` in the process environment before starting Streamlit, or enter a key in the sidebar for the current session. Do not commit keys to the project.

## Data handling

- Headline responses are cached in memory for ten minutes.
- Saved articles are written locally to `saved_articles.json` beside `main.py`.
- The app sends requests only when loading an RSS feed or NewsAPI results. It does not download linked articles, images, or videos.

## Limits

- RSS feed availability and content are controlled by their publishers.
- NewsAPI access, result volume, and rate limits depend on the selected key and plan.
- The dashboard does not validate whether a linked article remains available after it is saved.
