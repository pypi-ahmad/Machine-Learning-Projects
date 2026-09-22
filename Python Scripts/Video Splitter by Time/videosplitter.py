"""Split a local media file into a selected interval and its remaining tail."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def media_duration(input_path: Path, ffprobe: str) -> float:
    """Return the media duration in seconds from FFprobe."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def run_ffmpeg(command: list[str]) -> None:
    """Run FFmpeg and show its diagnostic output if it fails."""
    subprocess.run(command, check=True)


def split_media(input_path: Path, start: float, end: float, first_output: Path, second_output: Path) -> None:
    """Write the selected interval and the media after it without overwriting files."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg and ffprobe must be available on PATH.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input media does not exist: {input_path}")
    if start < 0 or end <= start:
        raise ValueError("Require 0 <= start < end.")
    duration = media_duration(input_path, ffprobe)
    if end > duration:
        raise ValueError(f"End time {end:g}s exceeds media duration {duration:.3f}s.")
    if first_output.exists() or second_output.exists():
        raise FileExistsError("Output files already exist; choose new output paths.")
    if first_output.resolve() == second_output.resolve() or input_path.resolve() in {first_output.resolve(), second_output.resolve()}:
        raise ValueError("Input and output paths must be different.")

    first_command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-n",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
        str(first_output),
    ]
    second_command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-n",
        "-ss",
        str(end),
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
        str(second_output),
    ]
    run_ffmpeg(first_command)
    run_ffmpeg(second_command)


def main() -> None:
    """Parse the split bounds and run FFmpeg."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input media path")
    parser.add_argument("start", type=float, help="selected interval start in seconds")
    parser.add_argument("end", type=float, help="selected interval end in seconds")
    parser.add_argument("first_output", type=Path, help="new file for the selected interval")
    parser.add_argument("second_output", type=Path, help="new file for media after the interval")
    args = parser.parse_args()
    try:
        split_media(args.input, args.start, args.end, args.first_output, args.second_output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Wrote {args.first_output} and {args.second_output}.")


if __name__ == "__main__":
    main()
