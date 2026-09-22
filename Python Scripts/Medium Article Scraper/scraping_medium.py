"""Save readable text from a public Medium article page."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).with_name("scraped_articles")


def medium_url(value: str) -> str:
    """Accept only HTTPS URLs hosted by Medium."""
    parsed = urlparse(value)
    is_medium = parsed.hostname == "medium.com" or (
        parsed.hostname is not None and parsed.hostname.endswith(".medium.com")
    )
    if parsed.scheme != "https" or not is_medium:
        raise argparse.ArgumentTypeError("URL must be an HTTPS medium.com URL")
    return value


def fetch_article(url: str) -> BeautifulSoup:
    """Download an article page with a stable user agent and timeout."""
    response = requests.get(
        url,
        headers={"User-Agent": "Medium-Article-Scraper/0.1"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_article(soup: BeautifulSoup, source_url: str) -> tuple[str, str]:
    """Return title and readable text from Medium's article or main element."""
    title_tag = soup.select_one("meta[property='og:title']") or soup.title
    title = title_tag.get("content", "").strip() if title_tag else ""
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    title = title or "Medium article"

    content = soup.find("article") or soup.find("main")
    if content is None:
        raise ValueError("No readable article content was found on this page.")
    text = content.get_text("\n", strip=True)
    if not text:
        raise ValueError("The article content was empty.")
    return title, f"Source: {source_url}\n\nTitle: {title}\n\n{text}\n"


def output_path(title: str, directory: Path) -> Path:
    """Build a Windows-safe text-file path from an article title."""
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._") or "medium_article"
    return directory / f"{filename}.txt"


def save_article(text: str, destination: Path, overwrite: bool) -> None:
    """Write UTF-8 text without replacing an existing file by default."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; rerun with --overwrite to replace it")
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    """Parse arguments, extract the article, and save it as text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", type=medium_url, help="HTTPS URL for a public medium.com article")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing title-matched file")
    args = parser.parse_args()

    try:
        title, text = extract_article(fetch_article(args.url), args.url)
        destination = output_path(title, args.output_dir)
        save_article(text, destination, args.overwrite)
    except (requests.RequestException, ValueError, FileExistsError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved article to {destination}")


if __name__ == "__main__":
    main()
