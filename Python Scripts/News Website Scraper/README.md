# News Website Scraper

Scrape the pagination links on Moneycontrol's technical-analysis news listing and save the extracted titles, dates, and image URLs as JSON.

## Requirements

- Python 3.14 or later
- Internet access to Moneycontrol

## Run

```powershell
uv sync --no-config
uv run --no-config python moneycontrol_scrapper.py --help
uv run --no-config python moneycontrol_scrapper.py
```

By default, output is written beside the script as `moneycontrol_<current date>.json`. Choose a different destination or listing page when needed:

```powershell
uv run --no-config python moneycontrol_scrapper.py --output .\headlines.json
uv run --no-config python moneycontrol_scrapper.py --url https://www.moneycontrol.com/news/technical-call-221.html
```

## Data and limits

- The scraper makes read-only HTTP requests and does not download linked articles or images.
- The JSON structure retains a separate list of titles, dates, and image URLs for each discovered page.
- The scraper uses a 15-second request timeout. It stops with an error if the expected Moneycontrol listing or pagination markup is unavailable.
- Website layout, availability, robots policy, and terms can change. Confirm that your use complies with the publisher's requirements before running it.
