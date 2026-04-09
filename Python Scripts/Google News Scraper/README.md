# Google News Scraper

> Scrape Google News articles by keyword and export results to an Excel spreadsheet.

## Overview

A command-line Python script that queries Google News RSS feed for a user-specified keyword, extracts article titles and links up to a requested count, and saves the results to an Excel (`.xlsx`) file using `pandas` and `openpyxl`.

## Features

- Searches Google News via RSS feed (`http://news.google.com/news?q=<term>&output=rss`)
- Extracts article titles and links from the XML response
- User-configurable search keyword and article count via interactive prompts
- Exports results to an Excel spreadsheet named `<keyword>_news_scrapper.xlsx`

## Project Structure

```
Google-News-Scraapper/
├── app.py                                  # Main script
├── cricket news_news_scrapper.xlsx         # Sample output file
├── requirements.txt                        # Python dependencies
└── README.md
```

## Requirements

- Python 3.x
- `requests`
- `pandas`
- `openpyxl`

From `requirements.txt`:
```
openpyxl==3.0.5
pandas==1.0.5
requests==2.24.0
```

## Installation

```bash
cd "Google-News-Scraapper"
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Interactive prompts:
```
Enter the news title keyword: cricket news
Enter the number of article count: 10
```

This generates a file named `cricket news_news_scrapper.xlsx` with columns `title` and `links`.

## How It Works

1. **`get_google_news_result(term, count)`** — Sends a GET request to Google News RSS endpoint with the search term. Parses the XML response using `xml.dom.minidom.parseString()`. Iterates over `<item>` elements, extracting `<title>` and `<link>` child node text. Returns two lists (titles, links) limited to the requested `count`.

2. **Main block** — Prompts user for a keyword and article count. Calls the scraping function, constructs a `pandas.DataFrame`, and exports it to Excel via `df.to_excel()`.

## Configuration

No configuration files needed. The search keyword and article count are provided interactively at runtime.

## Limitations

- **Google News RSS may be deprecated or restricted** — the `http://news.google.com/news?q=<term>&output=rss` endpoint may not return results in all regions or may be blocked.
- **Uses HTTP, not HTTPS** for the Google News URL.
- **No error handling** for network failures, empty responses, or invalid XML.
- **No command-line arguments** — only interactive input via `input()`.
- **XML parsing via `minidom`** — works but is verbose compared to alternatives like `feedparser`.
- The sample output file `cricket news_news_scrapper.xlsx` is included in the repo.

## Security Notes

No security concerns identified.

## License

Not specified.
