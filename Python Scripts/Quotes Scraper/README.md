# Quotes Scraper

Export quote text, authors, and tags from the public [quotes.toscrape.com](https://quotes.toscrape.com/) practice site to CSV.

## Setup

Requirements: Python 3.13+.

```powershell
cd "Quotes Scraper"
uv sync
```

## Usage

```powershell
uv run python quote_scraper.py --output quote_list.csv
```

Options:

- `--max-pages N`: Maximum pagination pages to fetch. Default: `10`, which covers the current practice-site dataset.
- `--overwrite`: Replace an existing CSV. Without it, an existing output file causes a clear error.

Tags are stored as semicolon-separated text in the `tags` column.

## Behavior

1. Fetches quote pages over HTTPS with a 20-second timeout.
2. Extracts quote text, author, and tags from each page.
3. Follows the site's next-page link until it ends or reaches `--max-pages`.
4. Writes UTF-8 CSV using standard CSV quoting.

The target is a public practice website. Its page markup or availability can change. This tool does not bypass access controls, rate limits, or CAPTCHA checks.

## Project files

```text
Quotes Scraper/
├── quote_scraper.py
├── Screenshot.png
├── pyproject.toml
└── uv.lock
```
