"""Record the full screen to an AVI file until Escape is pressed.

Usage:
    uv run python screen-recorder.py [--output recording.avi] [--fps 10]
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import ImageGrab


def default_output() -> Path:
    """Return a timestamped AVI filename in the current directory."""
    return Path(f"{time.time_ns() // 1_000_000}.avi")


def screenrecorder(output: Path, fps: float) -> None:
    """Record the primary display until the user presses Escape."""
    first_frame = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output file: {output}")

    try:
        frame = first_frame
        while True:
            cv2.imshow("Screen Recorder", frame)
            writer.write(frame)

            if cv2.waitKey(1) == 27:
                break

            frame = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
    finally:
        writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the primary screen to an AVI file.")
    parser.add_argument("--output", type=Path, default=default_output(), help="AVI output path")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    print(f"Recording to {args.output}. Press Escape in the preview window to stop.")
    screenrecorder(args.output, args.fps)


if __name__ == "__main__":
    main()
