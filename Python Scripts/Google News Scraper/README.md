# Google News Scraper

> Scrape Google News articles by keyword and export the results to an Excel spreadsheet.

## Overview

A command-line Python script that queries the Google News RSS feed for a keyword, extracts a requested number of article titles and links, and saves the results to an Excel (`.xlsx`) file with `pandas` and `openpyxl`.

## Features

- Searches Google News via its HTTPS RSS search feed
- Extracts article titles and links from the XML response
- User-configurable search keyword and article count through prompts or CLI options
- Exports results to an Excel spreadsheet named `<keyword>_news_scraper.xlsx`

## Project Structure

```
Google-News-Scraapper/
├── app.py                                  # Main script
├── cricket news_news_scrapper.xlsx         # Sample output file
├── pyproject.toml                          # Project metadata and dependencies
├── uv.lock                                 # Locked dependency versions
└── README.md
```

## Requirements

- Python 3.13+
- `requests`, `pandas`, and `openpyxl`, managed by uv in `pyproject.toml`

## Installation

```bash
cd "Google-News-Scraapper"
uv sync
```

## Usage

```bash
uv run python app.py
```

Interactive prompts:
```
Enter the news title keyword: cricket news
Enter the number of article count: 10
```

This generates a file named `cricket news_news_scraper.xlsx` with columns `title` and `links`.

Or provide the term, count, and output directly:

```bash
uv run python app.py "cricket news" --count 10 --output cricket-news.xlsx
```

## How it works

1. **`get_google_news_result(term, count)`** — Sends a bounded HTTPS request to the Google News RSS search endpoint and parses title/link pairs with Python's standard XML parser.

2. **Main block** — Gets a keyword and article count from arguments or prompts, refuses to overwrite existing output, and exports a `pandas.DataFrame` to Excel.

## Configuration

No configuration files needed. Use `--count` and `--output` to control the result count and destination.

## Limitations

- Google News RSS availability and result format can change.
- The script exports only the title and link supplied by RSS.
- The sample output file `cricket news_news_scrapper.xlsx` is included in the repo.

## Security Notes

No security concerns identified.

## License

Not specified.
