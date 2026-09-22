# YouTube Comment Scraper

Collect the comments currently loaded in a YouTube video page and save the visible author/comment pairs as UTF-8 CSV. The tool uses Selenium and requires an explicit non-dry-run invocation to open Chrome.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- A Selenium-supported Chrome installation

## Install

```powershell
cd "Python Scripts\YouTube Comment Scraper"
uv sync
```

Selenium manages a compatible driver when a supported browser is available.

## Preview

```powershell
uv run python .\webscrapindcomment.py "https://www.youtube.com/watch?v=VIDEO_ID" --dry-run
```

## Scrape currently loaded comments

```powershell
uv run python .\webscrapindcomment.py "https://www.youtube.com/watch?v=VIDEO_ID" --output .\comments.csv --scrolls 2
```

The browser closes after the scrape unless `--keep-open` is supplied.

## Notes

- The tool exports comments visible after the requested number of page-end scrolls; it does not guarantee complete comment coverage.
- YouTube can change its page structure, sign-in behavior, or automation restrictions, which may break the selectors.
- Scrape only content you are authorized to collect and handle comment data responsibly.
