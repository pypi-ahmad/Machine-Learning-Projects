"""Optionally apply an existing local image as the Windows wallpaper."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

SPI_SETDESKWALLPAPER = 20
SPIF_UPDATEINIFILE = 1
SPIF_SENDCHANGE = 2


def parse_args() -> argparse.Namespace:
    """Parse the local image and explicit apply flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Existing local wallpaper image")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Set the image as the Windows wallpaper",
    )
    return parser.parse_args()


def set_wallpaper(image_path: Path) -> None:
    """Set an existing image as the Windows desktop wallpaper."""
    if os.name != "nt":
        raise OSError("Setting the wallpaper is supported only on Windows.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    success = ctypes.windll.user32.SystemParametersInfoW(
        SPI_SETDESKWALLPAPER,
        0,
        str(image_path.resolve()),
        SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
    )
    if not success:
        raise ctypes.WinError()


def main() -> None:
    """Show or apply the selected local image."""
    args = parse_args()
    if not args.apply:
        print(f"Would set wallpaper to: {args.image.resolve()}")
        print("Pass --apply to change the desktop wallpaper.")
        return
    try:
        set_wallpaper(args.image)
    except (FileNotFoundError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error
    print("Wallpaper updated.")


if __name__ == "__main__":
    main()
