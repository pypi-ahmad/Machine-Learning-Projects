"""Save a public GeeksForGeeks article as a PDF through Chrome."""

import argparse
from base64 import b64decode
from pathlib import Path
from urllib.parse import urlparse

import requests
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

REQUEST_TIMEOUT = 15


def validate_article_url(url: str) -> None:
    """Reject non-GeeksForGeeks URLs before opening Chrome."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or '').lower()
    if parsed.scheme not in {'http', 'https'} or (
        hostname != 'geeksforgeeks.org' and not hostname.endswith('.geeksforgeeks.org')
    ):
        raise ValueError('Enter a valid GeeksForGeeks article URL.')


def download_article(url: str, output_path: Path) -> None:
    """Render an article in headless Chrome and save it as a PDF."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        pdf = driver.execute_cdp_cmd('Page.printToPDF', {'printBackground': True})
        output_path.write_bytes(b64decode(pdf['data']))
    finally:
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description='Download a GeeksForGeeks article as PDF.')
    parser.add_argument('url', nargs='?', help='GeeksForGeeks article URL')
    parser.add_argument('--output', type=Path, default=Path('article.pdf'), help='PDF output path')
    args = parser.parse_args()
    url = args.url or input('Provide GeeksForGeeks article URL: ').strip()

    try:
        validate_article_url(url)
        if args.output.exists():
            raise FileExistsError(f'Refusing to overwrite existing file: {args.output}')
        download_article(url, args.output)
    except (OSError, ValueError, WebDriverException, requests.RequestException) as error:
        parser.error(str(error))

    print(f'Article saved to {args.output}')


if __name__ == '__main__':
    main()
