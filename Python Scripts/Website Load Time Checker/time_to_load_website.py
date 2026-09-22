"""Measure the HTTP request and response-read time for one URL."""

from __future__ import annotations

import argparse
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def normalize_url(raw_url: str) -> str:
    """Return a valid HTTP(S) URL, adding HTTPS when no scheme is supplied."""
    url = raw_url.strip()
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provide a complete HTTP or HTTPS URL.")
    return url


def get_load_time(url: str, timeout: float = 10.0) -> float:
    """Return seconds spent requesting and reading a URL response."""
    if timeout <= 0:
        raise ValueError("Timeout must be greater than zero.")
    request = Request(url, headers={"User-Agent": "website-load-time-checker/1.0"})
    start = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        response.read()
    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    """Parse the target URL and request timeout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="HTTP(S) URL or hostname to measure")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the URL and show the request plan without connecting",
    )
    return parser.parse_args()


def main() -> None:
    """Run the load-time check."""
    args = parse_args()
    try:
        url = normalize_url(args.url)
        if args.dry_run:
            print(f"Would request {url} with a {args.timeout:g}-second timeout.")
            return
        elapsed = get_load_time(url, args.timeout)
    except (URLError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Request and response read completed in {elapsed:.3f} seconds.")


if __name__ == "__main__":
    main()
