# Twitter Scraper without API

> Scrape tweets by hashtag using snscrape and store them in a local SQLite database.

## Overview

A two-script project that scrapes Twitter tweets by hashtag without requiring API keys. `fetch_hashtags.py` uses the `snscrape` library to find tweets matching a given hashtag and stores them in a SQLite database. `display_hashtags.py` queries the database to display stored tweets filtered by hashtag.

## Features

- Scrape tweets by hashtag without Twitter API keys
- Configurable maximum number of tweets to fetch per query
- Persistent storage in a SQLite database (`TwitterDatabase.db`)
- Stores hashtag, username, tweet content, and URL for each tweet
- Search and display tweets from the database by hashtag
- Interactive loop — fetch or search multiple hashtags in one session

## Project Structure

```
Twitter_Scraper_without_API/
├── fetch_hashtags.py
├── display_hashtags.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- snscrape 0.3.4
- beautifulsoup4 4.9.3
- requests 2.25.1
- lxml 4.6.2
- PySocks 1.7.1
- Additional transitive dependencies in `requirements.txt`

## Installation

```bash
cd "Twitter_Scraper_without_API"
pip install -r requirements.txt
```

## Usage

### Fetching Tweets

```bash
python fetch_hashtags.py
```

1. Enter a hashtag (without `#`).
2. Enter the maximum number of tweets to fetch.
3. Tweets are stored in `TwitterDatabase.db`.
4. Press `y` to search another hashtag or any other key to exit.

### Displaying Tweets

```bash
python display_hashtags.py
```

1. Enter a hashtag to search (without `#`).
2. Matching tweets are displayed with username, content, and URL.
3. Press `y` to search again or any other key to exit.

## How It Works

1. **`fetch_hashtags.py`:**
   - Connects to (or creates) `TwitterDatabase.db` using `sqlite3`.
   - Creates a `tweets` table with columns: `HASHTAG`, `USERNAME`, `CONTENT`, `URL`.
   - Uses `snscrape.modules.twitter.TwitterSearchScraper` to iterate over tweets matching the hashtag.
   - Inserts each tweet's data into the database up to the user-specified maximum.

2. **`display_hashtags.py`:**
   - Connects to the same `TwitterDatabase.db`.
   - Fetches all rows from the `tweets` table.
   - Filters rows where the hashtag column matches the user's input.
   - Prints username, tweet content, and URL for each match.

## Configuration

- **Database path:** Hardcoded as `./Twitter_Scraper_without_API/TwitterDatabase.db` in both scripts. This assumes the scripts are run from the parent directory.

## Limitations

- The database path is relative and assumes execution from the parent directory of the project folder.
- `display_hashtags.py` fetches all rows and filters in Python instead of using a SQL `WHERE` clause — inefficient for large databases.
- No deduplication — running `fetch_hashtags.py` multiple times with the same hashtag creates duplicate entries.
- `snscrape` may break when Twitter changes its web interface.
- No error handling for network failures, invalid input, or database errors.

## License

Not specified.
