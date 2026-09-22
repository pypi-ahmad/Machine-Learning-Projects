"""Collect currently loaded YouTube trending cards into CSV or MongoDB."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import urlparse

from pymongo import MongoClient
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as expected
from selenium.webdriver.support.ui import WebDriverWait

DEFAULT_TRENDING_URL = "https://www.youtube.com/feed/trending"
FIELDNAMES = ["section", "title", "channel", "link", "views", "date"]


def parse_args() -> argparse.Namespace:
    """Parse output, browser, and optional MongoDB settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_TRENDING_URL, help="YouTube feed URL")
    parser.add_argument("--output", type=Path, default=Path("youtube_trending.csv"))
    parser.add_argument("--limit", type=int, default=10, help="Maximum cards to save (default: 10)")
    parser.add_argument("--scrolls", type=int, default=2, help="Page-end scrolls before extraction (default: 2)")
    parser.add_argument("--timeout", type=int, default=20, help="Browser wait timeout (default: 20)")
    parser.add_argument("--mongo-uri", help="Optional MongoDB URI for an additional copy")
    parser.add_argument("--database", default="youtube", help="MongoDB database name (default: youtube)")
    parser.add_argument("--keep-open", action="store_true", help="Leave Chrome open after scraping")
    parser.add_argument("--run", action="store_true", help="Open Chrome and collect data")
    return parser.parse_args()


def validate_youtube_url(value: str) -> str:
    """Validate an HTTP(S) YouTube URL."""
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provide a complete YouTube URL.")
    if hostname != "youtu.be" and not hostname.endswith("youtube.com"):
        raise ValueError("Provide a YouTube or youtu.be URL.")
    return value


def write_csv(records: list[dict[str, str]], output_path: Path) -> None:
    """Write scraped records to a UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def save_to_mongo(records: list[dict[str, str]], uri: str, database: str) -> None:
    """Store records in a MongoDB collection when explicitly configured."""
    if not records:
        return
    with MongoClient(uri, serverSelectionTimeoutMS=5_000) as client:
        client[database]["trending"].insert_many(records)


def card_record(card: object) -> dict[str, str] | None:
    """Extract one visible title, link, channel, and metadata line from a card."""
    try:
        title = card.find_element(By.CSS_SELECTOR, "#video-title")
        metadata = card.find_elements(By.CSS_SELECTOR, "#metadata-line span")
        channel = card.find_element(By.CSS_SELECTOR, "#channel-name a")
    except NoSuchElementException:
        return None
    metadata_text = [item.text for item in metadata if item.text]
    return {
        "section": "Trending",
        "title": title.text,
        "channel": channel.text,
        "link": title.get_attribute("href") or "",
        "views": metadata_text[0] if metadata_text else "",
        "date": metadata_text[1] if len(metadata_text) > 1 else "",
    }


def scrape_trending(url: str, limit: int, scrolls: int, timeout: int, keep_open: bool) -> list[dict[str, str]]:
    """Load the feed and collect records from currently rendered video cards."""
    if limit < 1 or scrolls < 0 or timeout < 1:
        raise ValueError("Limit and timeout must be positive; scrolls cannot be negative.")
    driver = webdriver.Chrome()
    try:
        driver.get(url)
        wait = WebDriverWait(driver, timeout)
        wait.until(expected.presence_of_element_located((By.TAG_NAME, "body")))
        for _ in range(scrolls):
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        cards = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer, ytd-rich-item-renderer")
        records = []
        for card in cards[:limit]:
            record = card_record(card)
            if record and record["title"]:
                records.append(record)
        return records
    finally:
        if not keep_open:
            driver.quit()


def main() -> None:
    """Preview or run a trending-feed scrape."""
    args = parse_args()
    try:
        url = validate_youtube_url(args.url)
        if not args.run:
            print(f"Would collect up to {args.limit} cards from {url} into {args.output.resolve()}")
            if args.mongo_uri:
                print(f"Would also write records to MongoDB database {args.database!r}.")
            print("Pass --run to open Chrome and collect data.")
            return
        records = scrape_trending(url, args.limit, args.scrolls, args.timeout, args.keep_open)
        write_csv(records, args.output)
        if args.mongo_uri:
            save_to_mongo(records, args.mongo_uri, args.database)
    except (OSError, ValueError, WebDriverException) as error:
        raise SystemExit(f"Scrape failed: {error}") from error
    print(f"Saved {len(records)} records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
