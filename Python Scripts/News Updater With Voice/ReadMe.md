# News Updater With Voice

Fetch NewsAPI headlines and read them through the Windows SAPI5 text-to-speech engine.

## Requirements

- Python 3.14 or later
- A Windows SAPI5 voice
- A NewsAPI key available as `NEWS_API_KEY`
- Internet access when fetching updates

## Run

```powershell
uv sync --no-config
$env:NEWS_API_KEY = "your-newsapi-key"
uv run --no-config python News.py
```

The default command reads one update: up to five `corona` headlines for India. Adjust the topic, country, and count as needed:

```powershell
uv run --no-config python News.py --query technology --country us --limit 3
```

To repeat an update every ten minutes, opt in explicitly:

```powershell
uv run --no-config python News.py --interval 600
```

Use `Ctrl+C` to stop a repeating update.

## Safety and limits

- `NEWS_API_KEY` is read only from the process environment and is never written to source or local configuration.
- The script fetches headline metadata from NewsAPI; it does not download linked articles.
- NewsAPI availability and result limits depend on the selected API plan.
- Speech output requires a working local Windows audio and SAPI5 setup.
