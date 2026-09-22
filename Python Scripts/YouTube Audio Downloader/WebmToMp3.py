"""Convert a local audio file to MP3 with MoviePy."""

from __future__ import annotations

import argparse
from pathlib import Path

from moviepy import AudioFileClip


def parse_args() -> argparse.Namespace:
    """Parse the source file and optional MP3 destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="Existing audio or video file")
    parser.add_argument("--output", type=Path, help="MP3 output path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the conversion plan without reading or writing media",
    )
    return parser.parse_args()


def output_path(input_file: Path, requested_output: Path | None) -> Path:
    """Return the requested output or an adjacent MP3 path."""
    return requested_output or input_file.with_suffix(".mp3")


def convert_to_mp3(input_file: Path, destination: Path) -> None:
    """Read one local media file and write an MP3 file."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with AudioFileClip(str(input_file)) as clip:
        clip.write_audiofile(str(destination))


def main() -> None:
    """Preview or run one media conversion."""
    args = parse_args()
    destination = output_path(args.input_file, args.output).resolve()
    if args.dry_run:
        print(f"Would convert {args.input_file.resolve()} to {destination}")
        return
    try:
        convert_to_mp3(args.input_file, destination)
    except (FileNotFoundError, OSError) as error:
        raise SystemExit(f"Conversion failed: {error}") from error
    print(f"Created MP3: {destination}")


if __name__ == "__main__":
    main()
