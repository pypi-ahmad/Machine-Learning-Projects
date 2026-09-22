"""Fetch details for the first matching IMDb feature film."""

from __future__ import annotations

import argparse
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


IMDB_URL = "https://www.imdb.com"
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = "Movie-Information-Scraper/0.1"
NOT_AVAILABLE = "Not available"


def fetch_soup(path: str, params: dict[str, str]) -> BeautifulSoup:
    """Fetch an IMDb page with a timeout and explicit user agent."""
    response = requests.get(
        urljoin(IMDB_URL, path),
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def first_title_url(soup: BeautifulSoup) -> str | None:
    """Return the first unique IMDb title link from a search result page."""
    for link in soup.select("a[href*='/title/']"):
        match = re.search(r"/title/(tt\d+)/", link.get("href", ""))
        if match:
            return urljoin(IMDB_URL, match.group())
    return None


def text_or_default(element: object) -> str:
    """Return normalized element text or a clear fallback."""
    return element.get_text(" ", strip=True) if element else NOT_AVAILABLE


def credit_names(soup: BeautifulSoup, label: str) -> list[str]:
    """Extract visible people from one labeled IMDb credit section."""
    for section in soup.select("li[data-testid='title-pc-principal-credit']"):
        heading = section.select_one("span")
        if heading and heading.get_text(" ", strip=True).lower().startswith(label):
            return [link.get_text(" ", strip=True) for link in section.select("a")]
    return []


def movie_details(soup: BeautifulSoup, url: str) -> dict[str, object]:
    """Extract available details from a current IMDb title page."""
    title = text_or_default(soup.select_one("h1"))
    year = text_or_default(soup.select_one("a[href*='releaseinfo']"))
    rating = text_or_default(
        soup.select_one("[data-testid='hero-rating-bar__aggregate-rating__score'] span")
    )
    runtime = text_or_default(soup.select_one("li[data-testid='title-techspec_runtime']"))
    genres = [genre.get_text(" ", strip=True) for genre in soup.select("a[href*='/search/title/?genres=']")]
    cast = [actor.get_text(" ", strip=True) for actor in soup.select("a[data-testid='title-cast-item__actor']")]
    plot = text_or_default(soup.select_one("[data-testid='plot-xl'], [data-testid='plot-l']"))
    return {
        "name": title,
        "year": year,
        "rating": rating,
        "runtime": runtime,
        "release_date": year,
        "genres": genres,
        "directors": credit_names(soup, "director"),
        "writers": credit_names(soup, "writer"),
        "cast": cast,
        "plot": plot,
        "url": url,
    }


def get_movie_details(movie_name: str) -> dict[str, object] | None:
    """Search IMDb for a feature film and return its available details."""
    search = fetch_soup("/search/title/", {"title": movie_name, "title_type": "feature"})
    url = first_title_url(search)
    if url is None:
        return None
    return movie_details(fetch_soup(url, {}), url)


def names(value: object) -> str:
    """Format a list field without treating a missing field as iterable text."""
    return ", ".join(value) if value else NOT_AVAILABLE


def print_movie(details: dict[str, object]) -> None:
    """Print a compact, readable movie report."""
    print(f"{details['name']} ({details['year']})")
    print(f"Rating: {details['rating']}")
    print(f"Runtime: {details['runtime']}")
    print(f"Release date: {details['release_date']}")
    print(f"Genres: {names(details['genres'])}")
    print(f"Directors: {names(details['directors'])}")
    print(f"Writers: {names(details['writers'])}")
    print(f"Cast: {names(details['cast'])}")
    print(f"Plot summary: {details['plot']}")
    print(f"IMDb: {details['url']}")


def main() -> None:
    """Parse a title query and print details for its first IMDb match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", nargs="+", help="movie title to search")
    args = parser.parse_args()
    try:
        details = get_movie_details(" ".join(args.movie))
    except requests.RequestException as error:
        raise SystemExit(f"IMDb request failed: {error}") from error
    if details is None:
        raise SystemExit("No matching IMDb feature film was found.")
    print_movie(details)


if __name__ == "__main__":
    main()
