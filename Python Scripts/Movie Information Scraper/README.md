# Movie Information Scraper

Command-line IMDb scraper that prints available details for the first matching feature film.

## Setup

Requirements: Python 3.13+.

```powershell
cd "Movie Information Scraper"
uv sync
```

## Usage

```powershell
uv run python movieInfoScraper.py "The Dark Knight"
```

The script searches IMDb feature-film results and prints the first matching title's available rating, runtime, genres, credits, cast, plot, and IMDb link.

## Behavior

1. Searches IMDb over HTTPS with a 20-second timeout.
2. Selects the first IMDb title result in the feature-film search.
3. Extracts fields from the current title page, using `Not available` for missing scalar fields.
4. Prints results to the console without writing files.

IMDb can change its markup, block or limit automated requests, or return incomplete records. This tool does not bypass access controls, rate limits, paywalls, or CAPTCHA checks. A title search can resolve to a different work with a similar name; inspect the printed IMDb link before relying on the result.

## Project files

```text
Movie Information Scraper/
├── movieInfoScraper.py
├── Screenshot.png
├── pyproject.toml
└── uv.lock
```
