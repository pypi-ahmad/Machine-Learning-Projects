"""Capture one or more local screenshots at a configurable interval."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


SECONDS_PER_UNIT = {"h": 3600, "m": 60, "s": 1}


def interval_seconds(unit: str, frequency: int) -> float:
    """Convert captures-per-unit into a minimum one-second interval."""
    if frequency < 1:
        raise ValueError("frequency must be at least 1")
    return max(1.0, SECONDS_PER_UNIT[unit] / frequency)


def capture_screenshot(directory: Path) -> Path:
    """Capture the current screen to a timestamped PNG file."""
    import pyautogui

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"screenshot_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    pyautogui.screenshot(str(destination))
    return destination


def main() -> None:
    """Parse capture options and run until the requested count is reached."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--path", type=Path, default=Path("images"), help="output directory")
    parser.add_argument("-t", "--unit", choices=SECONDS_PER_UNIT, default="h", help="h, m, or s")
    parser.add_argument("-f", "--frequency", type=int, default=1, help="captures per selected unit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", type=int, default=1, help="number of screenshots to capture")
    mode.add_argument("--continuous", action="store_true", help="capture until interrupted")
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("--count must be at least 1")
    try:
        interval = interval_seconds(args.unit, args.frequency)
        captured = 0
        while args.continuous or captured < args.count:
            destination = capture_screenshot(args.path)
            captured += 1
            print(f"Saved {destination}")
            if args.continuous or captured < args.count:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("Capture stopped by user.")
    except (OSError, ValueError) as error:
        raise SystemExit(f"Capture failed: {error}") from error


if __name__ == "__main__":
    main()
