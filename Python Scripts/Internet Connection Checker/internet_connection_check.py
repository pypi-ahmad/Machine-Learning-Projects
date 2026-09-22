"""Check whether a URL can be reached within a bounded timeout."""

from __future__ import annotations

import argparse

import requests


def internet_connection_test(url: str, timeout: float) -> bool:
    """Return whether the URL responds successfully within ``timeout`` seconds."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Could not reach {url}: {error}")
        return False
    print(f"Connected to {url} (HTTP {response.status_code}).")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check connectivity to an HTTPS URL.")
    parser.add_argument("--url", default="https://www.google.com/", help="HTTPS URL to check.")
    parser.add_argument("--timeout", type=float, default=10, help="Request timeout in seconds (default: 10).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero.")
    if not args.url.startswith("https://"):
        raise SystemExit("--url must use HTTPS.")
    raise SystemExit(0 if internet_connection_test(args.url, args.timeout) else 1)


if __name__ == "__main__":
    main()
