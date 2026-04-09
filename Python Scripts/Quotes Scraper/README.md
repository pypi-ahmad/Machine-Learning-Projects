# Scrape Quotes

> Scrapes all quotes from quotes.toscrape.com and saves them to a CSV file with quote text, author, and tags.

## Overview

A Python script that scrapes the website [quotes.toscrape.com](http://quotes.toscrape.com), traversing all pages via the "next" button, and collects every quote's text, author, and associated tags. The results are saved to a CSV file (`quote_list.csv`).

## Features

- Scrapes all quotes across all paginated pages automatically
- Extracts quote text, author name, and tags for each quote
- Saves results to `quote_list.csv` with headers: `quote`, `author`, `tags`
- Follows pagination ("next" button) until all pages are processed
- Uses BeautifulSoup for HTML parsing

## Project Structure

```
Scrape_quotes/
├── quote_scraper.py    # Main scraping script
├── requirements.txt    # Python dependencies
├── Screenshot.png      # Screenshot of output
└── README.md
```

## Requirements

- Python 3.x
- `beautifulsoup4`
- `requests` (v2.23.0)

## Installation

```bash
cd "Scrape_quotes"
pip install -r requirements.txt
```

## Usage

```bash
python quote_scraper.py
```

The script runs without any user input and generates `quote_list.csv` in the project directory.

**Sample CSV output:**

| quote | author | tags |
|-------|--------|------|
| "The world as we have created it..." | Albert Einstein | ['change', 'deep-thoughts', 'thinking', 'world'] |

## How It Works

1. Sends a GET request to `http://quotes.toscrape.com`.
2. Parses the HTML and finds all `<div class="quote">` elements.
3. For each quote, extracts:
   - Text from `<span class="text">`
   - Author from `<small class="author">`
   - Tags from all `<a class="tag">` elements (collected into a list)
4. Writes each quote as a row in `quote_list.csv` using `csv.DictWriter`.
5. Checks for a `<li class="next">` element; if found, follows the link to the next page and repeats. If not found, the loop ends.

## Configuration

- **Target URL**: Hardcoded to `http://quotes.toscrape.com`.
- **Output file**: `quote_list.csv` in the current working directory.

## Limitations

- The target URL is hardcoded — only works with `quotes.toscrape.com`.
- Tags are written as a Python list string representation (e.g., `['tag1', 'tag2']`) rather than a proper delimited format.
- Uses a bare `except` block that catches all exceptions, printing only `"Unknown Error!!!"` with no details.
- The CSV file variable `csv_file` could be referenced before assignment in the `finally` block if the `open()` call itself fails.
- No command-line arguments for output filename or URL customization.
- Overwrites `quote_list.csv` on each run without warning.

## Security Notes

No security concerns.

## License

Not specified.
