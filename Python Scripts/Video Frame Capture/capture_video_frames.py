"""Extract JPEG frames from a local video file."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_OUTPUT = Path("captured_frames")


def capture_frames(video_path: Path, output_directory: Path, every: int) -> int:
    """Save every selected frame and return the number of files written."""
    if output_directory.exists():
        raise FileExistsError(f"{output_directory} already exists; choose a new output directory.")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_number = 0
    saved = 0
    try:
        output_directory.mkdir(parents=True)
        while True:
            found, frame = video.read()
            if not found:
                break
            if frame_number % every == 0:
                output_path = output_directory / f"frame_{saved:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise OSError(f"Could not write frame: {output_path}")
                saved += 1
            frame_number += 1
    finally:
        video.release()
    return saved


def main() -> None:
    """Parse command-line arguments and extract video frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new frame directory")
    parser.add_argument("--every", type=int, default=1, help="save every Nth frame (default: 1)")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video does not exist or is not a file: {args.video}")
    if args.every < 1:
        parser.error("--every must be at least 1")

    try:
        saved = capture_frames(args.video, args.output, args.every)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved {saved} frame(s) to {args.output}.")


if __name__ == "__main__":
    main()
