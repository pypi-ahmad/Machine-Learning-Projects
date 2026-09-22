"""Scrape Moneycontrol technical-analysis listings into a JSON file."""

import argparse
import datetime
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


SOURCE_URL = "https://www.moneycontrol.com/news/technical-call-221.html"
REQUEST_TIMEOUT_SECONDS = 15


def get_soup(url: str) -> BeautifulSoup:
    """Fetch one page and return its parsed HTML."""
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": "news-website-scraper/1.0"},
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def scrape_page(url: str) -> list[dict[str, list[str]]]:
    """Extract titles, dates, and image URLs from one listing page."""
    category = get_soup(url).find("ul", {"id": "cagetory"})
    if category is None:
        raise ValueError("The expected Moneycontrol listing was not found.")

    images = category.find_all("img")
    return [
        {"title": list(map(lambda image: image.get("alt", "").strip(), images))},
        {"date": list(map(lambda item: item.get_text(strip=True), category.find_all("span")))},
        {"img_src": list(map(lambda image: image.get("src", "").strip(), images))},
    ]


def scrape_pages(url: str) -> dict[str, list[dict[str, list[str]]]]:
    """Follow the discovered pagination links and collect each page's data."""
    pagination = get_soup(url).find("div", attrs={"class": "pagenation"})
    if pagination is None:
        raise ValueError("The expected Moneycontrol pagination was not found.")

    anchors = pagination.find_all("a", href=re.compile(r"^((?!void).)*$"))
    links = list(
        map(
            lambda anchor: urljoin(url, anchor["href"]),
            filter(lambda anchor: anchor.get("href"), anchors),
        )
    )
    pages = map(scrape_page, tqdm(links, desc="Scraping pages"))
    return dict(map(lambda item: (str(item[0]), item[1]), enumerate(pages)))


def write_json(data: dict[str, list[dict[str, list[str]]]], output: Path) -> None:
    """Write scraper output as UTF-8 JSON."""
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    today = datetime.date.today().strftime("%B %d, %Y")
    parser = argparse.ArgumentParser(description="Scrape Moneycontrol technical-analysis news.")
    parser.add_argument("--url", default=SOURCE_URL, help="Listing page to scrape")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name(f"moneycontrol_{today}.json"),
        help="Destination JSON file",
    )
    args = parser.parse_args()

    try:
        write_json(scrape_pages(args.url), args.output)
    except (requests.RequestException, ValueError) as error:
        parser.exit(1, f"Scrape failed: {error}\n")
    print(f"Saved data to {args.output}")


if __name__ == "__main__":
    main()
