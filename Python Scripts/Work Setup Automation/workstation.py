"""Open a selected local application and work URLs after explicit confirmation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse
import webbrowser


def normalize_url(raw_url: str) -> str:
    """Return a valid HTTP(S) URL, adding HTTPS when needed."""
    url = raw_url.strip()
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid URL: {raw_url}")
    return url


def parse_args() -> argparse.Namespace:
    """Parse optional app and website launch targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="HTTP(S) URLs to open in the default browser")
    parser.add_argument("--app", type=Path, help="Existing local application to open")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Launch the app and URLs after showing their configured targets",
    )
    return parser.parse_args()


def launch(app_path: Path | None, urls: list[str]) -> None:
    """Open a local app and URLs through Windows and the default browser."""
    if app_path is not None:
        if os.name != "nt":
            raise OSError("Opening a local application is supported only on Windows.")
        if not app_path.is_file():
            raise FileNotFoundError(f"Application not found: {app_path}")
        os.startfile(str(app_path))
    for url in urls:
        webbrowser.open_new_tab(url)


def main() -> None:
    """Preview or open a work setup."""
    args = parse_args()
    try:
        urls = [normalize_url(url) for url in args.urls]
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if not args.app and not urls:
        raise SystemExit("Error: provide at least one URL or --app path.")
    if not args.open:
        if args.app:
            print(f"Would open application: {args.app}")
        if urls:
            print(f"Would open URLs: {', '.join(urls)}")
        print("Pass --open to launch these targets.")
        return

    try:
        launch(args.app, urls)
    except (FileNotFoundError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
