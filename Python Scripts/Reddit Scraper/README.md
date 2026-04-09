# Reddit Scraper without API

> Scrapes subreddit posts from old.reddit.com using BeautifulSoup and stores them in a SQLite database — no Reddit API key required.

## Overview

This project provides two scripts for collecting and viewing Reddit post data. `fetch_reddit.py` scrapes posts from any subreddit on `old.reddit.com`, following pagination to collect up to a user-specified number of entries, and stores the results in a SQLite database. `display_reddit.py` queries that database and displays stored posts for a given subreddit.

## Features

- Scrapes posts from any public subreddit via `old.reddit.com`
- Prompts for a sorting tag (hot, new, rising, controversial, top) stored as a label in the database (not applied to the scraping URL — the default "hot" page is always fetched)
- Collects post title, author, timestamp, upvotes, comment count, and URL
- Stores all scraped data in a local SQLite database (`SubredditDatabase.db`)
- Automatic pagination to collect multiple pages of posts
- Separate display script to query and view stored data
- User-agent header to mimic browser requests
- Interactive CLI for both scraping and viewing

## Project Structure

```
Reddit_Scraper_without_API/
├── fetch_reddit.py      # Scrapes subreddit posts and stores in SQLite
├── display_reddit.py    # Queries and displays stored posts from the database
├── requirements.txt     # Python dependencies
└── README.md
```

## Requirements

- Python 3.x
- `beautifulsoup4` (v4.9.3)
- `requests` (v2.25.1)
- `sqlite3` (Python standard library)

## Installation

```bash
cd "Reddit_Scraper_without_API"
pip install -r requirements.txt
```

## Usage

**Scraping posts:**

```bash
python fetch_reddit.py
```

You will be prompted to enter:
1. The subreddit name (e.g., `python`)
2. Maximum number of posts to collect
3. A sorting tag (hot / new / rising / controversial / top)

The scraper collects posts and stores them in `SubredditDatabase.db`. After completion, you can choose to scrape another subreddit or exit.

**Viewing stored posts:**

```bash
python display_reddit.py
```

Enter a subreddit name to search the database. Matching posts are displayed with their tag, title, author, timestamp, upvotes, comment count, and URL.

## How It Works

1. **fetch_reddit.py** constructs a URL like `https://old.reddit.com/r/{subreddit}/` and sends a GET request with a browser user-agent header. The sorting tag is stored in the database but is not appended to the URL.
2. It parses the HTML using BeautifulSoup, finding all `<div class="thing">` elements to extract post metadata (title, author, timestamp, upvotes, comments, URL).
3. Results are inserted into a SQLite table (`posts`) with columns: SUBREDDIT, TAG, TITLE, AUTHOR, TIMESTAMP, UPVOTES, COMMENTS, URL.
4. The scraper follows the "next" button link to paginate through results, sleeping 2 seconds between page requests.
5. **display_reddit.py** connects to the same database file and fetches all rows, filtering by subreddit name.

## Configuration

- The scraper targets `old.reddit.com` (hardcoded in `fetch_reddit.py`).
- The database file is `SubredditDatabase.db` in the working directory.
- A 2-second delay is used between pagination requests (`time.sleep(2)`).

## Limitations

- The sorting tag (hot, new, rising, controversial, top) is stored in the database but never appended to the scraping URL — the default page is always fetched regardless of tag selection.
- Relies on `old.reddit.com` HTML structure; any redesign will break the scraper.
- Posts where author, upvotes, or comments are missing are silently skipped via a bare `except AttributeError`.
- The display script fetches all rows from the database and filters in Python rather than using a SQL `WHERE` clause, which is inefficient for large datasets.
- Post URLs are prefixed with `www.reddit.com` but lack the `https://` scheme.
- No duplicate detection — scraping the same subreddit multiple times creates duplicate entries.
- The bare `except` on the pagination block catches all exceptions silently.

## Security Notes

- Uses a hardcoded user-agent string to mimic browser traffic; Reddit may rate-limit or block the scraper.
- No sensitive credentials are stored in the code.

## License

Not specified.



