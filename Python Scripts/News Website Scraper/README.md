# News Website Scraper

> A Python web scraper that extracts financial news (titles, dates, and images) from MoneyControl.com and saves the data as a JSON file.

## Overview

This script scrapes the technical analysis news section of MoneyControl.com. It navigates through all paginated result pages, extracts news titles, publication dates, and image URLs, and stores the collected data in a date-stamped JSON file organized by page number.

## Features

- Scrapes the MoneyControl technical analysis news section
- Automatic pagination — follows all "next page" links
- Extracts news titles, publication dates, and image source URLs
- Saves output as a JSON file named with the current date (e.g., `moneycontrol_March 03, 2026.json`)
- Progress bar using `tqdm` during scraping
- Data organized by page number in the output

## Project Structure

```
News_website_scraper/
├── moneycontrol_scrapper.py    # Main scraper script
├── README.md                   # Project documentation
└── images/                     # Screenshots for documentation
    ├── home.JPG                # MoneyControl home page screenshot
    ├── nextpage.JPG            # Pagination buttons screenshot
    ├── README.md
    └── result.JPG              # Sample JSON output screenshot
```

## Requirements

- Python 3.x
- requests
- beautifulsoup4
- lxml
- tqdm

## Installation

```bash
cd News_website_scraper
pip install requests beautifulsoup4 lxml tqdm
```

## Usage

```bash
python moneycontrol_scrapper.py
```

The script runs automatically and produces a JSON file in the current directory named `moneycontrol_<current_date>.json`.

## How It Works

1. **`setup(url)`**: Fetches the main page (`https://www.moneycontrol.com/news/technical-call-221.html`), finds the pagination div (`class="pagenation"`), extracts all next-page links (filtering out JavaScript void links), and calls `scrap()` on each page.

2. **`scrap(url, idx)`**: For each page:
   - Fetches the page HTML and parses it with `lxml`
   - Finds the `<ul id="cagetory">` element
   - Extracts `<span>` elements for dates
   - Extracts `<img>` elements — uses the `alt` attribute for titles and `src` for image URLs
   - Stores data in a `defaultdict(list)` keyed by page index

3. **`json_dump(data)`**: Writes the collected `defaultdict` to a JSON file named with today's date.

### Output Format

```json
{
  "0": [
    {"title": ["News Title 1", "News Title 2", ...]},
    {"date": ["Date 1", "Date 2", ...]},
    {"img_src": ["url1", "url2", ...]}
  ],
  "1": [...]
}
```

## Configuration

| Setting | Location | Value | Description |
|---------|----------|-------|-------------|
| `src_url` | Line 11 | `https://www.moneycontrol.com/news/technical-call-221.html` | Starting URL for scraping |

## Limitations

- Hardcoded to scrape only MoneyControl's technical analysis section — not configurable for other sections
- Relies on specific HTML structure (`class="pagenation"`, `id="cagetory"`, `<img>` alt attributes) — will break if MoneyControl redesigns
- No error handling for network failures or missing elements
- No rate limiting or request delays between pages
- The output JSON structure uses separate lists for titles, dates, and images rather than grouping them per article
- The `lxml` parser is required but not listed in a `requirements.txt` file
- No command-line arguments for customizing output path or source URL

## License

Not specified.
