# Reddit Scraper

Single-file CLI that archives visible public posts from `old.reddit.com` into a local SQLite database and displays saved posts.

## Setup

Requirements: Python 3.13+.

```powershell
cd "Reddit Scraper"
uv sync
```

## Usage

Fetch public posts:

```powershell
uv run python fetch_reddit.py fetch python --sort new --max-posts 25
```

Show saved posts:

```powershell
uv run python fetch_reddit.py show python
```

Options:

- `--database PATH`: SQLite database path. Default: `SubredditDatabase.db` beside the script.
- `fetch --sort`: One of `hot`, `new`, `rising`, `controversial`, or `top`.
- `fetch --max-posts`: Maximum visible posts to collect. Default: `25`.
- `fetch --delay`: Seconds between page requests. Default: `2.0`.

## Behavior

1. Validates the subreddit name and requested sort.
2. Fetches public old.reddit.com listing pages with a 20-second timeout.
3. Paginates until it reaches the requested post count or no next page exists.
4. Stores title, author, timestamp, score, comment text, and URL with parameterized SQLite queries.
5. Avoids duplicate rows for the same subreddit, sort, and post URL.

Reddit can change old-site markup, restrict automated requests, or return removed content. This tool does not authenticate, bypass rate limits, CAPTCHA checks, access controls, or paywalls. Use collected public data responsibly and according to Reddit's terms.

## Project files

```text
Reddit Scraper/
├── fetch_reddit.py
├── pyproject.toml
└── uv.lock
```
