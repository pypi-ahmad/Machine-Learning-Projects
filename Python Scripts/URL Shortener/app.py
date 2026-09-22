"""Shorten URLs with TinyURL.

Usage:
    uv run python app.py https://example.com --dry-run
    uv run python app.py https://example.com
"""

import argparse
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen


TINYURL_API = "https://tinyurl.com/api-create.php"


def build_request_url(url: str) -> str:
    """Validate an absolute HTTP(S) URL and build the TinyURL API request."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must include an http:// or https:// scheme and host.")
    return f"{TINYURL_API}?{urlencode({'url': url})}"


def short_url(url: str) -> str:
    """Request a shortened URL from TinyURL."""
    request_url = build_request_url(url)
    try:
        with urlopen(request_url, timeout=10) as response:
            return response.read().decode("utf-8").strip()
    except OSError as error:
        raise ValueError(f"TinyURL request failed: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Shorten HTTP(S) URLs with TinyURL.")
    parser.add_argument("urls", nargs="+", metavar="URL", help="One or more HTTP(S) URLs")
    parser.add_argument("--dry-run", action="store_true", help="Validate URLs without contacting TinyURL")
    args = parser.parse_args()

    for url in args.urls:
        try:
            request_url = build_request_url(url)
            if args.dry_run:
                print(f"Valid URL: {url}\nTinyURL request: {request_url}")
            else:
                print(short_url(url))
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
