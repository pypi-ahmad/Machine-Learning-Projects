"""Fetch and display the HTTP status for a URL."""

import argparse
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

REQUEST_TIMEOUT = 15


def check_url(url: str) -> tuple[int, str]:
    """Return the response status and reason for a valid HTTP(S) URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('URL must include an http:// or https:// scheme.')
    with urlopen(url, timeout=REQUEST_TIMEOUT) as response:
        return response.status, response.reason


def main() -> None:
    parser = argparse.ArgumentParser(description='Check an HTTP(S) URL status code.')
    parser.add_argument('url', nargs='?', help='URL to check')
    args = parser.parse_args()
    url = args.url or input('Enter the URL to be invoked: ').strip()

    try:
        status, reason = check_url(url)
    except HTTPError as error:
        print(f'Status: {error.code} 👎')
        print(f'Message: Request failed. Request returned reason - {error.reason}')
    except (URLError, ValueError) as error:
        print('Status: unavailable 👎')
        print(f'Message: Request failed. {error}')
    else:
        print(f'Status code: {status} 👍')
        print(f'Message: Request succeeded. Request returned message - {reason}')


if __name__ == '__main__':
    main()
