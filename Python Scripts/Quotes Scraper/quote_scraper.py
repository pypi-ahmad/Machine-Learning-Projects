"""Export quotes, authors, and tags from quotes.toscrape.com to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://quotes.toscrape.com/"
REQUEST_TIMEOUT_SECONDS = 20
FIELDS = ("quote", "author", "tags")


def fetch_page(url: str) -> BeautifulSoup:
    """Download one quotes page with a bounded request."""
    response = requests.get(
        url,
        headers={"User-Agent": "Quotes-Scraper/0.1"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_page(soup: BeautifulSoup, page_url: str) -> tuple[list[dict[str, str]], str | None]:
    """Extract quote records and the next-page URL from one page."""
    records: list[dict[str, str]] = []
    for quote in soup.select("div.quote"):
        text = quote.select_one("span.text")
        author = quote.select_one("small.author")
        if text is not None and author is not None:
            tags = ";".join(map(lambda tag: tag.get_text(" ", strip=True), quote.select("a.tag")))
            records.append(
                {
                    "quote": text.get_text(" ", strip=True),
                    "author": author.get_text(" ", strip=True),
                    "tags": tags,
                }
            )
    next_link = soup.select_one("li.next a[href]")
    next_url = urljoin(page_url, next_link["href"]) if next_link else None
    return records, next_url


def scrape_quotes(max_pages: int) -> list[dict[str, str]]:
    """Fetch and parse no more than ``max_pages`` quote pages."""
    url: str | None = BASE_URL
    records: list[dict[str, str]] = []
    page_count = 0
    while url is not None and page_count < max_pages:
        page_records, url = parse_page(fetch_page(url), url)
        records.extend(page_records)
        page_count += 1
    return records


def write_csv(records: list[dict[str, str]], output: Path, overwrite: bool) -> None:
    """Write records as UTF-8 CSV without overwriting by default."""
    mode = "w" if overwrite else "x"
    with output.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Fetch quotes and save them to a CSV file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("quote_list.csv"))
    parser.add_argument("--max-pages", type=int, default=10, help="maximum pages to fetch")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output CSV")
    args = parser.parse_args()
    if args.max_pages < 1:
        parser.error("--max-pages must be at least 1")
    try:
        write_csv(scrape_quotes(args.max_pages), args.output, args.overwrite)
    except (requests.RequestException, FileExistsError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved quotes to {args.output}")


if __name__ == "__main__":
    main()
