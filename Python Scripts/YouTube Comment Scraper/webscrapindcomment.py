"""Collect currently loaded YouTube comments into a local CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import urlparse

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as expected
from selenium.webdriver.support.ui import WebDriverWait


def parse_args() -> argparse.Namespace:
    """Parse the video URL and scrape limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", type=Path, default=Path("comments.csv"))
    parser.add_argument("--scrolls", type=int, default=2, help="Number of page-end scrolls (default: 2)")
    parser.add_argument("--timeout", type=int, default=20, help="Wait timeout in seconds (default: 20)")
    parser.add_argument("--keep-open", action="store_true", help="Leave Chrome open after scraping")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the plan without opening Chrome or requesting YouTube",
    )
    return parser.parse_args()


def validate_youtube_url(value: str) -> str:
    """Validate an HTTP(S) YouTube video URL."""
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provide a complete YouTube URL.")
    if hostname != "youtu.be" and not hostname.endswith("youtube.com"):
        raise ValueError("Provide a YouTube or youtu.be URL.")
    return value


def write_comments(comments: list[dict[str, str]], output_path: Path) -> None:
    """Write collected author/comment pairs as UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Author", "Comment"])
        writer.writeheader()
        writer.writerows(comments)


def scrape_comments(url: str, scrolls: int, timeout: int, keep_open: bool) -> list[dict[str, str]]:
    """Open a video, scroll for comments, and return currently loaded pairs."""
    if scrolls < 0 or timeout < 1:
        raise ValueError("Scrolls cannot be negative and timeout must be at least one.")
    driver = webdriver.Chrome()
    try:
        driver.get(url)
        wait = WebDriverWait(driver, timeout)
        wait.until(expected.presence_of_element_located((By.TAG_NAME, "body")))
        for _ in range(scrolls):
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        authors = wait.until(expected.presence_of_all_elements_located((By.CSS_SELECTOR, "#author-text")))
        comments = driver.find_elements(By.CSS_SELECTOR, "#content-text")
        return [
            {"Author": author.text, "Comment": comment.text}
            for author, comment in zip(authors, comments)
            if author.text or comment.text
        ]
    finally:
        if not keep_open:
            driver.quit()


def main() -> None:
    """Preview or scrape comments from one YouTube video."""
    args = parse_args()
    try:
        url = validate_youtube_url(args.url)
        if args.dry_run:
            print(f"Would collect currently loaded comments from {url} into {args.output.resolve()}")
            return
        comments = scrape_comments(url, args.scrolls, args.timeout, args.keep_open)
        write_comments(comments, args.output)
    except (OSError, ValueError, WebDriverException) as error:
        raise SystemExit(f"Scrape failed: {error}") from error
    print(f"Saved {len(comments)} comments to {args.output.resolve()}")


if __name__ == "__main__":
    main()
