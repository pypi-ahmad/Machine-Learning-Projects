# X post scraper

`fetch_hashtags.py` fetches one bounded page of recent X posts through the X API
v2 and stores them in a local SQLite database. The same command can list posts
already saved in that database.

## Requirements

- Python 3.11 or later
- An X developer account with access to recent search
- A bearer token available to the process as `X_BEARER_TOKEN`

The token is read at runtime. It is not stored in source code or the database.
Set it in your Windows environment, then restart the terminal or editor that
runs the script.

## Install

```powershell
cd "Python Scripts/Twitter Scraper"
uv sync
```

## Fetch posts

Use an X API v2 query. Hashtag and language operators belong in that query.

```powershell
uv run python fetch_hashtags.py fetch "#python lang:en" --limit 10
```

`--limit` must be between 10 and 100 because the script makes one recent-search
request. Each new post is stored once by post ID in `twitter_posts.db` beside the
script.

Use `--database` before the command to select another local database:

```powershell
uv run python fetch_hashtags.py --database D:\data\posts.db fetch "from:example" --limit 25
```

## List saved posts

```powershell
uv run python fetch_hashtags.py list
uv run python fetch_hashtags.py list --query "#python lang:en"
```

The optional `--query` filter matches the exact query used when posts were
saved.

## Database notes

The project writes a `posts` table with the post ID, source query, author ID,
text, and creation time. Any older `tweets` table is left unchanged; this script
does not infer a conversion from the old schema.

Search availability, historical range, and rate limits depend on the account's
X API access level. This tool fetches recent results only and does not paginate.
