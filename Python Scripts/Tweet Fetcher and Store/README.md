# Fetch and Store Tweets

> A Python script that fetches tweets matching a specific hashtag using the Twitter API (via Tweepy) and stores them in a CSV file.

## Overview

This script authenticates with the Twitter API using OAuth credentials, searches for tweets matching a configured hashtag, and appends the results (timestamp and text) to a CSV file called `tweets.csv`.

## Features

- Authenticates with the Twitter API via OAuth 1.0a using Tweepy
- Searches for tweets by hashtag with configurable parameters
- Filters tweets by language (English) and start date
- Handles Twitter API rate limits automatically (`wait_on_rate_limit=True`)
- Exports tweet timestamp and text to a CSV file
- Fetches up to 100 tweets per API request page

## Project Structure

```
Fetch_and_store_tweets/
├── fetch_store_tweet.py   # Main script for fetching and storing tweets
├── requirements.txt       # Python dependencies
└── img/                   # Documentation images for setup guide
```

## Requirements

- Python 3.x
- `tweepy==3.9.0`
- A Twitter Developer account with API keys

## Installation

```bash
cd "Fetch_and_store_tweets"
pip install -r requirements.txt
```

## Usage

1. Configure your Twitter API credentials and hashtag in `fetch_store_tweet.py` (see Configuration below).
2. Run:

```bash
python fetch_store_tweet.py
```

3. Tweets will be printed to the console and appended to `tweets.csv` in the same directory.

## How It Works

1. Creates an OAuth handler with `consumer_key` and `consumer_secret`.
2. Sets the access token using `access_token` and `access_token_secret`.
3. Initializes the Tweepy API client with rate-limit waiting enabled.
4. Opens (or creates) `tweets.csv` in append mode.
5. Uses `tweepy.Cursor` to paginate through `api.search` results for the configured hashtag.
6. For each tweet, prints the creation date and text, then writes a row to the CSV (date + UTF-8 encoded text).

## Configuration

The following variables must be set in `fetch_store_tweet.py`:

| Variable              | Description                          |
|-----------------------|--------------------------------------|
| `consumer_key`        | Twitter API consumer key             |
| `consumer_secret`     | Twitter API consumer secret          |
| `access_token`        | Twitter API access token             |
| `access_token_secret` | Twitter API access token secret      |
| `hastag`              | Hashtag to search for (note: variable is misspelled as `hastag`) |

Additional hardcoded search parameters:
- `count=100` — tweets per page
- `lang="en"` — English tweets only
- `since="2017-04-03"` — tweets since this date

## Limitations

- Uses `api.search` which is part of the Twitter v1.1 API — this has been deprecated in favor of the v2 API
- The `since` date is hardcoded to `"2017-04-03"`
- The CSV file is opened in append mode (`'a'`), so re-running adds duplicate tweets
- The variable `hastag` is misspelled (should be `hashtag`)
- Tweet text is encoded as UTF-8 bytes in the CSV, which may cause display issues
- No limit on the number of tweets fetched (iterates until exhausted)
- Tweepy 3.9.0 is outdated; newer versions have breaking changes

## Security Notes

- **Twitter API credentials are stored as empty strings** in the source code — these must be filled in before use and should ideally be stored in environment variables or a separate config file
- Never commit real API keys to version control

## License

Not specified.
