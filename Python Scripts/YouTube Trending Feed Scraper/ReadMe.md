# YouTube Trending Feed Scraper

Collect currently rendered video cards from a YouTube trending feed and save them as CSV. MongoDB persistence is optional and requires an explicit URI.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- A Selenium-supported Chrome installation for scraping
- MongoDB only when `--mongo-uri` is used

## Install

```powershell
cd "Python Scripts\YouTube Trending Feed Scraper"
uv sync
```

## Preview a scrape

The default command is a preview; it does not open Chrome or request YouTube:

```powershell
uv run python .\youtube_scrapper.py
```

## Collect trending cards

```powershell
uv run python .\youtube_scrapper.py --run --output .\youtube_trending.csv --limit 10 --scrolls 2
```

Use another YouTube feed URL when appropriate:

```powershell
uv run python .\youtube_scrapper.py --url "https://www.youtube.com/feed/trending" --run
```

To additionally write the records to MongoDB, pass an explicit connection URI:

```powershell
uv run python .\youtube_scrapper.py --run --mongo-uri "mongodb://localhost:27017" --database youtube
```

## Read saved data

```powershell
uv run python .\scrap_reader.py --csv .\youtube_trending.csv
```

Or read from the configured MongoDB collection:

```powershell
uv run python .\scrap_reader.py --mongo-uri "mongodb://localhost:27017" --database youtube
```

## Notes

- This extracts only cards currently rendered after the requested scrolls, not a guaranteed complete trending feed.
- YouTube can change layout, availability, or automation restrictions, which may break selectors.
- Collect only content you are authorized to access and handle exported data responsibly.
