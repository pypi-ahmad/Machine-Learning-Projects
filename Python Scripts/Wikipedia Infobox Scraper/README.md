# Wikipedia Infobox Scraper

> A Tkinter GUI application that scrapes and displays Wikipedia infobox data for any search query.

## Overview

This application provides a graphical interface where users can type a search term. It constructs a Wikipedia URL, fetches the page, parses the HTML infobox table using BeautifulSoup, and displays the key–value pairs from the infobox in a popup window.

## Features

- Tkinter-based GUI with a search entry field and submit button
- Constructs Wikipedia URLs by capitalizing and joining search words with underscores
- Scrapes the `<table class="infobox">` element from Wikipedia pages
- Extracts all `<th>` / `<td>` pairs from the infobox into a dictionary
- Displays results in a color-coded popup window (cyan4 keys, cyan2 values)
- Error popup dialog when fetching or parsing fails
- HTTP status code validation (checks for 200 response)

## Project Structure

```
Wiki_Infobox_Scraper/
├── main.py              # Main application script
├── requirements.txt     # Pinned dependencies
└── README.md
```

## Requirements

- Python 3.x
- `beautifulsoup4` (4.9.3)
- `requests` (2.25.1)
- `tkinter` (included with standard Python)

All dependencies are pinned in `requirements.txt`.

## Installation

```bash
cd "Wiki_Infobox_Scraper"
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. A window titled "Wikipedia Infobox" appears with a search field
2. Type a search query (e.g., "Python programming language") and click **Submit**
3. A popup window displays the infobox data as key–value pairs
4. If the page has no infobox or the request fails, an error popup appears

## How It Works

1. **URL Construction** — The user's input is split into words, each capitalized, and joined with underscores to form a Wikipedia URL path (e.g., `"python programming"` → `https://en.wikipedia.org/wiki/Python_Programming`).
2. **HTML Parsing** — `requests.get()` fetches the page, then `BeautifulSoup` finds the first `<table>` with class `infobox`.
3. **Data Extraction** — Iterates over all `<tr>` rows; for each row with a `<th>`, stores `th.text → td.text` in a dictionary.
4. **Display** — Creates a `Toplevel` popup window with `Label` widgets for each key–value pair.

## Configuration

No configuration files. The Wikipedia base URL (`https://en.wikipedia.org/wiki/`) is hardcoded in `main.py`.

## Limitations

- Only scrapes English Wikipedia (`en.wikipedia.org`)
- URL construction capitalizes each word, which may not match Wikipedia's actual page title for all queries
- Uses bare `except` clauses that silently swallow parsing errors
- The `info_dict` global is reused across queries; it's cleared after each successful scrape but could accumulate on errors
- No support for disambiguation pages
- The error popup only shows "ERROR FETCHING DATA" with no details
- Font `'Century Schoolbook L'` may not be available on all systems

## Security Notes

No security concerns identified.

## License

Not specified.
