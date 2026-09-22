"""Search Google News RSS and export article titles and links to Excel."""

import argparse
from pathlib import Path
import re
from xml.etree import ElementTree

import pandas as pd
import requests

RSS_URL = 'https://news.google.com/rss/search'
REQUEST_TIMEOUT = 15


def get_google_news_result(term: str, count: int) -> list[dict[str, str]]:
    """Return up to count title/link pairs from Google News RSS."""
    response = requests.get(RSS_URL, params={'q': term}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    articles = []
    for item in root.findall('.//item')[:count]:
        articles.append({'title': item.findtext('title', ''), 'links': item.findtext('link', '')})
    return articles


def default_output_path(term: str) -> Path:
    """Create a Windows-safe default filename from a search term."""
    safe_term = re.sub(r'[<>:"/\\|?*]+', '_', term).strip(' ._') or 'google_news'
    return Path(f'{safe_term}_news_scraper.xlsx')


def main() -> None:
    parser = argparse.ArgumentParser(description='Export Google News RSS results to Excel.')
    parser.add_argument('term', nargs='?', help='Search term')
    parser.add_argument('--count', type=int, default=10, help='Maximum article count')
    parser.add_argument('--output', type=Path, help='New Excel output file')
    args = parser.parse_args()
    term = args.term or input('Enter the news title keyword: ').strip()
    if not term:
        parser.error('A search term is required')
    if args.count < 1:
        parser.error('--count must be at least 1')

    output_path = args.output or default_output_path(term)
    if output_path.exists():
        parser.error(f'Refusing to overwrite existing file: {output_path}')

    try:
        articles = get_google_news_result(term, args.count)
    except (ElementTree.ParseError, requests.RequestException) as error:
        parser.error(f'News search failed: {error}')
    if not articles:
        parser.error('No articles were returned for this search.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(articles, columns=['title', 'links']).to_excel(output_path, index=False)
    print(f'Saved {len(articles)} articles to {output_path}')


if __name__ == '__main__':
    main()
