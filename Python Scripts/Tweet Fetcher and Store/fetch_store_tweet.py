"""Fetch recent X posts through the v2 API and save them as CSV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import tweepy


DEFAULT_OUTPUT = Path("tweets.csv")
FIELDNAMES = ("id", "created_at", "text")


def bearer_token() -> str:
    """Return the X API token without placing credentials in source code."""
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise SystemExit("Error: X_BEARER_TOKEN is required. Set it and relaunch the host if needed.")
    return token


def fetch_posts(client: tweepy.Client, query: str, limit: int) -> list[tweepy.Tweet]:
    """Fetch one bounded page of recent posts matching a v2 search query."""
    response = client.search_recent_tweets(
        query=query,
        max_results=limit,
        tweet_fields=["created_at"],
    )
    return response.data or []


def write_posts(posts: list[tweepy.Tweet], output: Path, append: bool) -> int:
    """Write posts as UTF-8 CSV rows without silently overwriting a file."""
    if not posts:
        return 0
    if output.exists() and not append:
        raise FileExistsError(f"{output} already exists; use --append or choose another path.")
    mode = "a" if append else "x"
    with output.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not append or file.tell() == 0:
            writer.writeheader()
        writer.writerows(posts)
    return len(posts)


def main() -> None:
    """Fetch a bounded set of recent matching posts and write a CSV file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help='X v2 query, for example "#python lang:en"')
    parser.add_argument("--limit", type=int, default=10, help="posts to request (10-100; default: 10)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV path (default: tweets.csv)")
    parser.add_argument("--append", action="store_true", help="append to an existing CSV")
    args = parser.parse_args()
    if not 10 <= args.limit <= 100:
        parser.error("--limit must be between 10 and 100")

    try:
        posts = fetch_posts(tweepy.Client(bearer_token=bearer_token(), wait_on_rate_limit=True), args.query, args.limit)
        count = write_posts(posts, args.output, args.append)
    except (OSError, tweepy.TweepyException) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Saved {count} post(s) to {args.output}.")


if __name__ == "__main__":
    main()
