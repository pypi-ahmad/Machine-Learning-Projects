"""Save one random English Wikipedia article as readable text."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup


RANDOM_URL = "https://en.wikipedia.org/wiki/Special:Random"
REQUEST_TIMEOUT_SECONDS = 20


def fetch_random_article() -> BeautifulSoup:
    """Request one random article with an identifiable user agent and timeout."""
    response = requests.get(
        RANDOM_URL,
        headers={"User-Agent": "Random-Wikipedia-Article/0.1"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_article(soup: BeautifulSoup) -> str:
    """Return title and non-empty article paragraphs as UTF-8-ready text."""
    heading = soup.select_one("h1")
    paragraphs = soup.select("#mw-content-text p") or soup.select("p")
    body = "\n\n".join(map(lambda paragraph: paragraph.get_text(" ", strip=True), paragraphs))
    if heading is None or not body:
        raise ValueError("Wikipedia did not return readable article content.")
    return f"{heading.get_text(' ', strip=True)}\n\n{body}\n"


def save_article(text: str, output: Path, overwrite: bool) -> None:
    """Write text without replacing an existing output file by default."""
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8") as file:
        file.write(text)


def main() -> None:
    """Fetch one random article and save it to a local text file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("random_wiki.txt"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args()
    try:
        save_article(extract_article(fetch_random_article()), args.output, args.overwrite)
    except (requests.RequestException, FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved article to {args.output}")


if __name__ == "__main__":
    main()
