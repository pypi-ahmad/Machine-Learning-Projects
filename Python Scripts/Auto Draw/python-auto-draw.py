"""Plan or draw a rectangular spiral with PyAutoGUI."""

from __future__ import annotations

import argparse
import time


def spiral_segments(distance: int, decrement: int) -> list[tuple[int, int]]:
    """Return relative mouse movements for a shrinking rectangular spiral."""
    if distance <= 0 or decrement <= 0:
        raise ValueError("Distance and decrement must be greater than zero.")
    segments: list[tuple[int, int]] = []
    while distance > 0:
        segments.append((distance, 0))
        distance -= decrement
        if distance <= 0:
            break
        segments.extend(((0, distance), (-distance, 0)))
        distance -= decrement
        if distance > 0:
            segments.append((0, -distance))
    return segments


def draw(segments: list[tuple[int, int]], delay: float, duration: float) -> None:
    """Wait for focus, then draw the supplied relative movements."""
    import pyautogui

    pyautogui.FAILSAFE = True
    print(f"Drawing starts in {delay:g} seconds. Move the mouse to a screen corner to abort.")
    time.sleep(delay)
    pyautogui.click()
    for horizontal, vertical in segments:
        pyautogui.dragRel(horizontal, vertical, duration=duration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=250)
    parser.add_argument("--decrement", type=int, default=5)
    parser.add_argument("--delay", type=float, default=10)
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--draw", action="store_true", help="perform mouse actions; default is preview")
    args = parser.parse_args()
    try:
        segments = spiral_segments(args.distance, args.decrement)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Planned {len(segments)} drawing segments.")
    if args.draw:
        draw(segments, args.delay, args.duration)
    else:
        print("Preview only. Re-run with --draw to control the mouse.")


if __name__ == "__main__":
    main()
