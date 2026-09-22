"""Fetch public old.reddit.com posts into SQLite or display saved posts."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://old.reddit.com"
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_DATABASE = Path(__file__).with_name("SubredditDatabase.db")
SORTS = ("hot", "new", "rising", "controversial", "top")


def subreddit_name(value: str) -> str:
    """Validate Reddit's basic public subreddit-name format."""
    normalized = value.removeprefix("r/").lower()
    if not normalized.replace("_", "").isalnum() or not 3 <= len(normalized) <= 21:
        raise argparse.ArgumentTypeError("subreddit must contain 3-21 letters, numbers, or underscores")
    return normalized


def fetch_page(url: str) -> BeautifulSoup:
    """Request one old.reddit.com page with a timeout and clear user agent."""
    response = requests.get(
        url,
        headers={"User-Agent": "Reddit-Scraper/0.1 (public post archiver)"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_posts(soup: BeautifulSoup, subreddit: str, sort: str) -> tuple[list[tuple[str, ...]], str | None]:
    """Extract visible post fields and the next old.reddit.com URL."""
    records: list[tuple[str, ...]] = []
    for post in soup.select("div.thing"):
        title = post.select_one("a.title")
        author = post.select_one("a.author")
        timestamp = post.select_one("time")
        comments = post.select_one("a.comments")
        if title is not None:
            records.append(
                (
                    subreddit,
                    sort,
                    title.get_text(" ", strip=True),
                    author.get_text(" ", strip=True) if author else "[deleted]",
                    timestamp.get("datetime", "") if timestamp else "",
                    post.get("data-score", ""),
                    comments.get_text(" ", strip=True) if comments else "",
                    urljoin("https://www.reddit.com", title.get("href", "")),
                )
            )
    next_link = soup.select_one("span.next-button a[href]")
    return records, urljoin(BASE_URL, next_link["href"]) if next_link else None


def scrape_posts(subreddit: str, sort: str, max_posts: int, delay_seconds: float) -> list[tuple[str, ...]]:
    """Collect no more than ``max_posts`` visible posts from one sorted listing."""
    url: str | None = f"{BASE_URL}/r/{subreddit}/{sort}/"
    records: list[tuple[str, ...]] = []
    while url is not None and len(records) < max_posts:
        page_records, url = parse_posts(fetch_page(url), subreddit, sort)
        records.extend(page_records)
        if url is not None and len(records) < max_posts:
            time.sleep(delay_seconds)
    return records[:max_posts]


def connect_database(path: Path) -> sqlite3.Connection:
    """Open the database and ensure its deduplicated posts table exists."""
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            subreddit TEXT NOT NULL,
            tag TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            upvotes TEXT NOT NULL,
            comments TEXT NOT NULL,
            url TEXT NOT NULL,
            UNIQUE(subreddit, tag, url)
        )
        """
    )
    return connection


def store_posts(connection: sqlite3.Connection, records: list[tuple[str, ...]]) -> int:
    """Insert unseen records and return SQLite's changed-row count."""
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records
    )
    connection.commit()
    return connection.total_changes - before


def saved_posts(connection: sqlite3.Connection, subreddit: str) -> list[tuple[str, ...]]:
    """Return one subreddit's saved records using a SQL filter."""
    return connection.execute(
        "SELECT tag, title, author, timestamp, upvotes, comments, url FROM posts WHERE subreddit = ? ORDER BY timestamp DESC",
        (subreddit,),
    ).fetchall()


def main() -> None:
    """Run the fetch or show command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="scrape and save public posts")
    fetch.add_argument("subreddit", type=subreddit_name)
    fetch.add_argument("--sort", choices=SORTS, default="hot")
    fetch.add_argument("--max-posts", type=int, default=25)
    fetch.add_argument("--delay", type=float, default=2.0, help="seconds between page requests")
    show = commands.add_parser("show", help="display saved posts")
    show.add_argument("subreddit", type=subreddit_name)
    args = parser.parse_args()

    if args.command == "fetch" and (args.max_posts < 1 or args.delay < 0):
        parser.error("--max-posts must be positive and --delay cannot be negative")
    try:
        with connect_database(args.database) as connection:
            if args.command == "fetch":
                inserted = store_posts(
                    connection,
                    scrape_posts(args.subreddit, args.sort, args.max_posts, args.delay),
                )
                print(f"Saved {inserted} new post(s) for r/{args.subreddit}.")
            else:
                rows = saved_posts(connection, args.subreddit)
                for row in rows:
                    print("\n".join(row), end="\n\n")
                print(f"Displayed {len(rows)} post(s) for r/{args.subreddit}.")
    except (requests.RequestException, sqlite3.Error, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
