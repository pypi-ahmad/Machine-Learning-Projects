"""Download a random Unsplash wallpaper and optionally apply it on Windows."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import requests

SPI_SETDESKWALLPAPER = 20
SPIF_UPDATEINIFILE = 1
SPIF_SENDCHANGE = 2
UNSPLASH_RANDOM_URL = "https://api.unsplash.com/photos/random"


def parse_args() -> argparse.Namespace:
    """Parse download and wallpaper options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="wallpaper", help="Unsplash search query")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "wallpaper.jpg",
        help="Downloaded image path (default: .\\wallpaper.jpg)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Set the downloaded image as the Windows wallpaper",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned operation without calling Unsplash or changing wallpaper",
    )
    return parser.parse_args()


def unsplash_access_key() -> str:
    """Return the required Unsplash key without exposing it."""
    access_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("UNSPLASH_ACCESS_KEY is required to download a wallpaper.")
    return access_key


def random_wallpaper_url(access_key: str, query: str) -> str:
    """Request a random landscape wallpaper URL from Unsplash."""
    response = requests.get(
        UNSPLASH_RANDOM_URL,
        params={
            "client_id": access_key,
            "query": query,
            "orientation": "landscape",
        },
        timeout=30,
    )
    response.raise_for_status()
    try:
        return response.json()["urls"]["full"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Unsplash did not return an image URL.") from error


def download_image(image_url: str, output_path: Path) -> Path:
    """Download an image to the requested path."""
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


def set_wallpaper(image_path: Path) -> None:
    """Set an existing image as the Windows desktop wallpaper."""
    if os.name != "nt":
        raise OSError("Setting the wallpaper is supported only on Windows.")
    success = ctypes.windll.user32.SystemParametersInfoW(
        SPI_SETDESKWALLPAPER,
        0,
        str(image_path.resolve()),
        SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
    )
    if not success:
        raise ctypes.WinError()


def main() -> None:
    """Run the wallpaper downloader."""
    args = parse_args()
    output_path = args.output.resolve()
    if args.dry_run:
        action = "download and apply" if args.apply else "download"
        print(f"Would {action} a {args.query!r} wallpaper at {output_path}")
        return

    try:
        image_path = download_image(
            random_wallpaper_url(unsplash_access_key(), args.query), output_path
        )
        print(f"Downloaded wallpaper: {image_path}")
        if args.apply:
            set_wallpaper(image_path)
            print("Wallpaper updated.")
    except (OSError, requests.RequestException, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
