"""Download the best available audio stream from one YouTube URL."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


def parse_args() -> argparse.Namespace:
    """Parse the YouTube URL and local output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "downloads",
        help="Directory for the downloaded audio stream (default: .\\downloads)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the URL and show the destination without contacting YouTube",
    )
    return parser.parse_args()


def validate_youtube_url(value: str) -> str:
    """Validate an HTTP(S) URL with a YouTube hostname."""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provide a complete YouTube URL.")
    hostname = parsed.hostname or ""
    if hostname != "youtu.be" and not hostname.endswith("youtube.com"):
        raise ValueError("Provide a YouTube or youtu.be URL.")
    return value


def download_audio(url: str, output_dir: Path) -> None:
    """Download the best available audio-only stream with yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])


def main() -> None:
    """Validate and optionally download one audio stream."""
    args = parse_args()
    try:
        url = validate_youtube_url(args.url)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
    output_dir = args.output_dir.resolve()
    if args.dry_run:
        print(f"Would download the best audio stream from {url} into {output_dir}")
        return
    try:
        download_audio(url, output_dir)
    except yt_dlp.utils.DownloadError as error:
        raise SystemExit(f"Download failed: {error}") from error


if __name__ == "__main__":
    main()
