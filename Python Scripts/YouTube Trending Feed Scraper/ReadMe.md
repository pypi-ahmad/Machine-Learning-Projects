# YouTube Trending Feed Scraper

> Scrape the top 10 trending videos from each YouTube trending category and save the data to CSV or MongoDB.

## Overview

This project consists of two scripts: a **scraper** that uses Selenium to extract trending video data from YouTube's four trending categories (Now, Music, Gaming, Movies), and a **reader** that displays previously saved data from either a CSV file or a MongoDB database.

## Features

- Scrapes YouTube trending pages across 4 categories: Now, Music, Gaming, Movies
- Extracts the top 10 videos per category (40 total)
- Captures video title, channel, link, views, and date for each entry
- Saves data to **CSV** (`Youtube.csv`), **MongoDB** (`Youtube.trending`), or both
- Separate reader script to display saved data interactively
- Command-line flags (`-c` for CSV, `-m` for MongoDB) for both scripts
- MongoDB schema defined using `mongoengine` with index on `section`
- Interactive section-by-section display in the reader

## Project Structure

```
Youtube Trending Feed Scrapper/
├── youtube_scrapper.py    # Selenium-based scraper script
├── scrap_reader.py        # Data reader/display script
└── README.md
```

## Requirements

- Python 3.x
- `selenium`
- `pandas`
- `pymongo`
- `mongoengine`
- ChromeDriver executable (same version as your Chrome browser)
- MongoDB Community Server (required only for `-m` flag)

## Installation

```bash
cd "Youtube Trending Feed Scrapper"
pip install selenium pymongo mongoengine pandas
```

1. Download [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) matching your Chrome version
2. Place `chromedriver.exe` in the same directory as the scripts
3. Install [MongoDB Community Server](https://docs.mongodb.com/manual/administration/install-community/) (for MongoDB storage)

## Usage

### Scraping

```bash
# Save to CSV
python youtube_scrapper.py -c

# Save to MongoDB
python youtube_scrapper.py -m

# Save to both
python youtube_scrapper.py -c -m
```

### Reading saved data

```bash
# Read from CSV
python scrap_reader.py -c

# Read from MongoDB
python scrap_reader.py -m
```

The reader displays data section by section, prompting `Show Section? [y/n]` before each category. Enter `n` to exit.

**Note**: The reader requires exactly one flag (`-c` or `-m`, not both) — it uses an XNOR gate validation.

## How It Works

### Scraper (`youtube_scrapper.py`)
1. **`load_driver()`** — Initializes a Chrome WebDriver using `chromedriver.exe`.
2. **`page_scrap(driver)`** — Navigates to each of 4 trending page URLs, waits 3 seconds for page load, then extracts the first 10 `ytd-video-renderer` elements. Parses title, link, channel, views, and date from each card.
3. **`save_to_db()`** — Creates a `mongoengine.Document` instance and saves it to the `Youtube` MongoDB database.
4. **`append_to_df()` / `save_to_csv()`** — Appends records to a pandas DataFrame, then exports to `Youtube.csv`.

### Reader (`scrap_reader.py`)
1. **`read_mongo()`** — Connects to MongoDB at `127.0.0.1` and returns all documents from `Youtube.trending`.
2. **`read_csv()`** — Reads `Youtube.csv` using pandas and returns rows as a list.
3. **`display(data)`** — Iterates through records, printing a section header every 10 entries with a continue/exit prompt.

## Configuration

| Setting | Value | Location |
|---------|-------|----------|
| ChromeDriver path | `chromedriver.exe` (same directory) | `youtube_scrapper.py` |
| MongoDB host | `127.0.0.1` (default) | `scrap_reader.py` |
| MongoDB database | `Youtube` | Both scripts |
| MongoDB collection | `trending` | Both scripts |
| CSV filename | `Youtube.csv` | Both scripts |
| Page load delay | 3 seconds | `youtube_scrapper.py` |

## Limitations

- Uses deprecated Selenium methods (`find_elements_by_tag_name`, `find_elements_by_id`) — requires Selenium < 4.x or code updates
- The `pandas.DataFrame.append()` method is deprecated in newer pandas versions
- YouTube page structure changes may break the scraper's element selectors
- Hardcoded 3-second sleep instead of explicit waits
- ChromeDriver must be manually downloaded and version-matched
- The reader's XNOR validation means you cannot read from both CSV and MongoDB simultaneously
- Bare `except` clause when removing `"•"` character from metadata
- No headless mode option for the Chrome browser

## Security Notes

No credentials or API keys are used. MongoDB connects to localhost without authentication.

## License

Not specified.
