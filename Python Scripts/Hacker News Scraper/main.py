"""Fetch Hacker News listing pages and write plain-text summaries."""

import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://news.ycombinator.com/news"


def fetch_page(page: int) -> list[tuple[str, str]]:
    response = requests.get(BASE_URL, params={"p": page}, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.select(".titleline > a")
    return [(link.get_text(" ", strip=True), link.get("href", "")) for link in links]


def write_page(items: list[tuple[str, str]], page: int, output: Path) -> None:
    lines = [f"Hacker News page {page}", "=" * 40]
    for index, (title, url) in enumerate(items, start=1):
        lines.extend([f"{index}. {title}", url, ""])
    (output / f"NewsPage{page}.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Hacker News listing pages.")
    parser.add_argument("pages", type=int, help="Number of pages to fetch (1-20)")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args()
    if not 1 <= args.pages <= 20:
        parser.error("pages must be between 1 and 20")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for page in range(1, args.pages + 1):
        items = fetch_page(page)
        write_page(items, page, output)
        print(f"Wrote {len(items)} stories to NewsPage{page}.txt")


if __name__ == "__main__":
    main()
