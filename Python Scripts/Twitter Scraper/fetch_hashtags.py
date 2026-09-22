"""Fetch and list X posts in a local SQLite database."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import tweepy


DEFAULT_DATABASE = Path(__file__).with_name("twitter_posts.db")


def bearer_token() -> str:
    """Return the X API token without storing credentials in the project."""
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise SystemExit("Error: X_BEARER_TOKEN is required. Set it and relaunch the host if needed.")
    return token


def connect_database(path: Path) -> sqlite3.Connection:
    """Open the local database and create the modern posts table."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            author_id TEXT,
            text TEXT NOT NULL,
            created_at TEXT
        )
        """
    )
    return connection


def fetch_posts(client: tweepy.Client, query: str, limit: int) -> list[tweepy.Tweet]:
    """Fetch one bounded page of recent posts for an X API v2 query."""
    response = client.search_recent_tweets(
        query=query,
        max_results=limit,
        tweet_fields=["author_id", "created_at"],
    )
    return response.data or []


def store_posts(connection: sqlite3.Connection, query: str, posts: list[tweepy.Tweet]) -> int:
    """Store new posts and return how many IDs were not already present."""
    stored = 0
    for post in posts:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO posts (id, query, author_id, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (post.id, query, post.author_id, post.text, post.created_at.isoformat()),
        )
        stored += cursor.rowcount
    connection.commit()
    return stored


def list_posts(connection: sqlite3.Connection, query: str | None) -> list[sqlite3.Row]:
    """Return stored posts, optionally limited to an exact saved query."""
    if query:
        return connection.execute(
            "SELECT id, query, author_id, text, created_at FROM posts WHERE query = ? ORDER BY created_at DESC",
            (query,),
        ).fetchall()
    return connection.execute("SELECT id, query, author_id, text, created_at FROM posts ORDER BY created_at DESC").fetchall()


def print_posts(posts: list[sqlite3.Row]) -> None:
    """Print stored posts in a readable terminal format."""
    for post in posts:
        print(f"{post['id']} | {post['created_at']} | {post['author_id']}\n{post['text']}\nquery: {post['query']}\n")
    print(f"{len(posts)} post(s).")


def main() -> None:
    """Run the fetch or list command selected by the user."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="SQLite database path")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="fetch and save recent posts")
    fetch.add_argument("query", help='X v2 query, for example "#python lang:en"')
    fetch.add_argument("--limit", type=int, default=10, help="posts to request (10-100; default: 10)")
    listing = commands.add_parser("list", help="show stored posts")
    listing.add_argument("--query", help="show only an exact saved query")
    args = parser.parse_args()
    if args.command == "fetch" and not 10 <= args.limit <= 100:
        parser.error("--limit must be between 10 and 100")

    try:
        with connect_database(args.database) as connection:
            if args.command == "fetch":
                client = tweepy.Client(bearer_token=bearer_token(), wait_on_rate_limit=True)
                posts = fetch_posts(client, args.query, args.limit)
                print(f"Saved {store_posts(connection, args.query, posts)} new post(s).")
            else:
                print_posts(list_posts(connection, args.query))
    except (OSError, sqlite3.Error, tweepy.TweepyException) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
