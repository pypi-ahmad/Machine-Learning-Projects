# Fetch and store X posts

`fetch_store_tweet.py` runs one bounded X API v2 recent-search request and saves
the returned post ID, creation time, and text to a UTF-8 CSV file.

## Requirements

- Python 3.11 or later
- An X developer account with access to the recent-search endpoint
- A bearer token available to the process as `X_BEARER_TOKEN`

The token is read at runtime and is never stored in source code. Set it in your
Windows environment, then restart the terminal or editor that runs the script.

## Install

```powershell
cd "Python Scripts/Tweet Fetcher and Store"
uv sync
```

## Usage

Pass an X API v2 query. The service accepts its own query operators, so language
or hashtag filters belong in the query itself.

```powershell
uv run python fetch_store_tweet.py "#python lang:en" --limit 10 --output tweets.csv
```

`--limit` must be between 10 and 100, which is the supported size of one recent-
search result page. The script will not replace an existing CSV unless you pass
`--append`.

```powershell
uv run python fetch_store_tweet.py "from:example" --limit 25 --output example.csv --append
```

Run `uv run python fetch_store_tweet.py --help` for the full command reference.

## Output

The CSV has these columns:

```text
id,created_at,text
```

An empty result creates no CSV file and reports that zero posts were saved.

## Notes

This project uses X API v2 through current Tweepy releases. Search availability,
historical range, and rate limits depend on the account's X API access level.
The script only requests recent results and does not perform pagination.
