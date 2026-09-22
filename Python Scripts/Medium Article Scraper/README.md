# Medium Article Scraper

Save readable text from one public `medium.com` article page to a local UTF-8 text file.

## Setup

Requirements: Python 3.13+.

```powershell
cd "Medium Article Scraper"
uv sync
```

## Usage

```powershell
uv run python scraping_medium.py "https://medium.com/@author/article-slug"
```

By default, the output is saved under `scraped_articles/` beside the script. The filename is derived from the article title and is not overwritten by default.

Options:

- `--output-dir PATH`: Select another output directory.
- `--overwrite`: Replace an existing title-matched text file.

## Behavior

1. Validates that the supplied URL is HTTPS and hosted on `medium.com`.
2. Fetches the page with a 20-second timeout.
3. Extracts the page's Open Graph title and readable text from its `article` element, falling back to `main` where necessary.
4. Writes the source URL, title, and extracted text using UTF-8.

Medium can change its markup, restrict automated requests, or return incomplete content for paywalled articles. This tool does not bypass access controls, rate limits, paywalls, or CAPTCHA checks.

## Project files

```text
Medium Article Scraper/
├── scraping_medium.py
├── scraped_articles/
├── pyproject.toml
└── uv.lock
```
