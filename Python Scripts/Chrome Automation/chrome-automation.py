"""Preview or open a list of URLs in Google Chrome on Windows."""

import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit
import webbrowser


DEFAULT_URLS = (
    "stackoverflow.com",
    "github.com/avinashkranjan",
    "gmail.com",
    "google.co.in",
    "youtube.com",
)


def normalize_url(value: str) -> str:
    """Return an HTTP(S) URL suitable for opening in a browser."""
    url = value.strip()
    if not url:
        raise ValueError("URLs cannot be empty.")
    if not urlsplit(url).scheme:
        url = f"https://{url}"
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid URL: {value}")
    return url


def find_chrome() -> Path | None:
    """Return a common Windows Chrome executable path, if installed."""
    roots = (
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    )
    for root in roots:
        if root:
            candidate = Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
            if candidate.is_file():
                return candidate
    return None


def open_urls(urls: tuple[str, ...], chrome_path: Path) -> None:
    """Open normalized URLs in tabs through the selected Chrome executable."""
    browser = webbrowser.get(f'"{chrome_path}" %s')
    for url in urls:
        print(f"Opening: {url}")
        browser.open_new_tab(url)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview or open URLs in Google Chrome.")
    parser.add_argument("urls", nargs="*", help="URLs to open; defaults to the built-in list.")
    parser.add_argument("--open", action="store_true", help="Open URLs instead of printing a preview.")
    parser.add_argument("--chrome-path", type=Path, help="Path to chrome.exe.")
    args = parser.parse_args()

    raw_urls = tuple(args.urls) if args.urls else DEFAULT_URLS
    try:
        urls = tuple(map(normalize_url, raw_urls))
    except ValueError as error:
        parser.error(str(error))

    if not args.open:
        print("Preview only. Pass --open to launch Chrome:")
        for url in urls:
            print(f"  {url}")
        return 0

    chrome_path = args.chrome_path or find_chrome()
    if chrome_path is None or not chrome_path.is_file():
        parser.error("Chrome was not found. Supply --chrome-path C:\\path\\to\\chrome.exe")

    try:
        open_urls(urls, chrome_path)
    except webbrowser.Error as error:
        parser.error(f"Unable to launch Chrome: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
