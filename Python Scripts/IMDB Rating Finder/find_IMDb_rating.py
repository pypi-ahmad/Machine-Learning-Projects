"""Find IMDb ratings and genres for titles derived from local filenames."""

import argparse
import os
from pathlib import Path
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests

SEARCH_URL = 'https://www.imdb.com/search/title/'
REQUEST_TIMEOUT = 15
USER_AGENT = 'IMDbRatingFinder/1.0 (local metadata utility)'


def film_titles(directory: Path) -> list[str]:
    """Return non-empty filename stems from files directly inside a directory."""
    if not directory.is_dir():
        raise ValueError(f'Film directory not found: {directory}')
    titles = []
    for name in sorted(os.listdir(directory)):
        candidate = directory / name
        if candidate.is_file() and candidate.stem:
            titles.append(candidate.stem)
    return titles


def parse_search_result(html: str, title: str) -> dict[str, str] | None:
    """Return the first legacy IMDb search result matching a title, if present."""
    soup = BeautifulSoup(html, 'html.parser')
    for result in soup.select('.lister-item-content'):
        name_node = result.select_one('h3 a')
        if name_node is None or title.casefold() not in name_node.get_text(strip=True).casefold():
            continue
        rating_node = result.select_one('.inline-block.ratings-imdb-rating[data-value]')
        genre_node = result.select_one('.genre')
        return {
            'Film Name': name_node.get_text(strip=True),
            'Rating': rating_node.get('data-value', 'Unavailable') if rating_node else 'Unavailable',
            'Genre': genre_node.get_text(' ', strip=True) if genre_node else 'Unavailable',
        }
    return None


def find_rating(session: requests.Session, title: str) -> dict[str, str] | None:
    """Request IMDb's title search page and parse a matching result."""
    response = session.get(
        SEARCH_URL,
        params={'title': title},
        headers={'User-Agent': USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return parse_search_result(response.text, title)


def main() -> None:
    parser = argparse.ArgumentParser(description='Export IMDb ratings for local film filenames.')
    parser.add_argument('directory', type=Path, help='Directory containing film files')
    parser.add_argument('--output', type=Path, default=Path('film_ratings.csv'), help='New CSV output path')
    parser.add_argument('--delay', type=float, default=0.5, help='Seconds between IMDb requests')
    args = parser.parse_args()
    if args.delay < 0:
        parser.error('--delay must not be negative')
    if args.output.exists():
        parser.error(f'Refusing to overwrite existing file: {args.output}')

    try:
        titles = film_titles(args.directory)
    except ValueError as error:
        parser.error(str(error))
    if not titles:
        parser.error('No files were found in the film directory.')

    records = []
    with requests.Session() as session:
        for title in titles:
            try:
                record = find_rating(session, title)
            except requests.RequestException as error:
                print(f'Skipped {title}: {error}')
            else:
                if record is None:
                    print(f'No IMDb result found for {title}')
                else:
                    records.append(record)
            time.sleep(args.delay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=['Film Name', 'Rating', 'Genre']).to_csv(
        args.output, index=False, encoding='utf-8'
    )
    print(f'Saved {len(records)} rating(s) to {args.output}')


if __name__ == '__main__':
    main()
