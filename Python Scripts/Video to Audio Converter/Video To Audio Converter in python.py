"""Download a YouTube audio stream without changing its container extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytube
from pytube import YouTube


def parse_args() -> argparse.Namespace:
    """Parse the video URL and local output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for the downloaded audio stream (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the URL shape and show the planned destination without downloading",
    )
    return parser.parse_args()


def video_id_from_url(video_url: str) -> str:
    """Return a YouTube video ID or raise a clear error for an invalid URL."""
    video_id = pytube.extract.video_id(video_url)
    if not video_id:
        raise ValueError("The URL does not contain a YouTube video ID.")
    return video_id


def download_audio(video_url: str, output_dir: Path) -> Path:
    """Download the first available audio-only stream and return its real path."""
    stream = YouTube(video_url).streams.filter(only_audio=True).first()
    if stream is None:
        raise RuntimeError("No audio-only stream is available for this video.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(stream.download(output_path=str(output_dir)))


def main() -> None:
    """Run the command-line downloader."""
    args = parse_args()
    video_url = args.url or input("Enter YouTube video URL: ").strip()
    try:
        video_id = video_id_from_url(video_url)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    destination = args.output_dir.resolve()
    if args.dry_run:
        print(f"Would download the audio stream for {video_id} into {destination}")
        return

    try:
        downloaded_file = download_audio(video_url, destination)
    except Exception as error:
        raise SystemExit(f"Download failed: {error}") from error
    print(f"Downloaded audio stream: {downloaded_file}")


if __name__ == "__main__":
    main()
