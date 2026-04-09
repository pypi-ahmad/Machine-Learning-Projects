# Scrape Hacker News

> Scrapes article listings from Hacker News and saves structured article information to text files.

## Overview

A Python script that fetches one or more pages from Hacker News (`news.ycombinator.com`), extracts article metadata (rank, title, source, URL, author, score, posting time), and saves the results to text files in a local `HackerNews/` directory. Supports up to 20 pages and optional verbose output.

## Features

- Scrapes article listings from Hacker News by page number
- Extracts rank, title, source website, URL, author, score, and posting time for each article
- Saves results to formatted text files (`HackerNews/NewsPage{N}.txt`)
- Supports scraping up to 20 pages in a single run
- Optional verbose mode to display progress during scraping
- Automatically creates the `HackerNews/` output directory
- Handles connection errors and invalid input gracefully
- Uses `SoupStrainer` to parse only `<td>` elements for efficiency

## Project Structure

```
Scrape_Hacker_News/
├── main.py       # Main script with scraping logic and CLI
└── README.md
```

## Requirements

- Python 3.x
- `requests`
- `beautifulsoup4`

## Installation

```bash
cd "Scrape_Hacker_News"
pip install requests beautifulsoup4
```

## Usage

```bash
python main.py
```

**Prompts:**
1. Enter the number of pages to fetch (1–20).
2. Choose verbose output (`y` or `n`).

**Example:**
```
Enter number of pages that you want the HackerNews for (max 20): 3
Want verbose output y/[n] ? y
Fetching Page 1...
Fetching Page 2...
Fetching Page 3...
```

Output files are saved to `HackerNews/NewsPage1.txt`, `HackerNews/NewsPage2.txt`, etc.

## How It Works

1. The user specifies how many pages to scrape (capped at 20) and whether to enable verbose output.
2. For each page, a GET request is sent to `https://news.ycombinator.com/?p={page_no}`.
3. The HTML response is parsed with BeautifulSoup using a `SoupStrainer` that only processes `<td>` elements.
4. Article data is extracted from `<td class="title">` (rank and title) and `<td class="subtext">` (score, time, author) elements.
5. For each article, the script extracts rank, title text, source site, URL (prefixed with the HN domain if relative), author, score, and posting time.
6. Results are written to a formatted text file with clear separators between articles.
7. Connection errors and request exceptions are caught and reported per page.

## Configuration

- **Maximum pages**: Capped at 20 in the code (`page_no = min(page_no, 20)`).
- **Output directory**: `HackerNews/` in the current working directory (created automatically).

## Limitations

- Relies on Hacker News HTML class names (`storylink`, `rank`, `score`, `hnuser`, `sitestr`, `age`); HN redesigns will break the scraper.
- Articles with missing metadata fields get fallback "Could not get..." text rather than being skipped.
- No rate limiting between page requests (only the page cap limits load).
- URLs that don't start with `https` are prefixed with `https://news.ycombinator.com/`, which may not always be correct.
- Overwrites existing output files without warning.
- No option to specify output format (always text files).

## Security Notes

No security concerns.

## License

Not specified.


