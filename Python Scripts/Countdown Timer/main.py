"""Countdown Timer — CLI tool.

Counts down from a given duration and plays a terminal bell when done.
Supports hours:minutes:seconds input or simple seconds.

Usage:
    python main.py
    python main.py 5m
    python main.py 1h30m
    python main.py 90          # 90 seconds
"""

import sys
import time


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_duration(text: str) -> int:
    """Parse strings like '5m', '1h30m', '90', '1:30:00' into total seconds."""
    text = text.strip().lower()
    if not text:
        raise ValueError("Empty duration.")

    # HH:MM:SS or MM:SS
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        else:
            raise ValueError("Use H:MM:SS or MM:SS format.")

    # e.g. 1h30m45s
    seconds = 0
    current = ""
    for ch in text:
        if ch.isdigit():
            current += ch
        elif ch == "h":
            seconds += int(current or 0) * 3600
            current = ""
        elif ch == "m":
            seconds += int(current or 0) * 60
            current = ""
        elif ch == "s":
            seconds += int(current or 0)
            current = ""
        else:
            raise ValueError(f"Unexpected character: '{ch}'")
    if current:
        seconds += int(current)  # bare number = seconds

    if seconds <= 0:
        raise ValueError("Duration must be greater than zero.")
    return seconds


def format_duration(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

def run_countdown(total_seconds: int, label: str = "") -> None:
    print(f"\n  Timer started: {format_duration(total_seconds)}" +
          (f"  [{label}]" if label else ""))
    print("  (Press Ctrl+C to cancel)\n")

    start = time.monotonic()
    try:
        for remaining in range(total_seconds, -1, -1):
            elapsed = time.monotonic() - start
            # Stay in sync with wall clock
            target = total_seconds - remaining
            drift = elapsed - target
            sleep_time = max(0.0, 1.0 - drift)

            display = format_duration(remaining)
            print(f"\r  ⏱  {display}  ", end="", flush=True)

            if remaining == 0:
                break
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n\n  Timer cancelled.")
        return

    # Bell + finish message
    print(f"\r  ✓  Time's up! \a", flush=True)
    if label:
        print(f"  [{label}] complete.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_duration_from_user() -> tuple[int, str]:
    while True:
        raw = input("  Enter duration (e.g. 5m, 1h30m, 90, 0:05:00): ").strip()
        try:
            seconds = parse_duration(raw)
            label = input("  Label (optional, press Enter to skip): ").strip()
            return seconds, label
        except ValueError as e:
            print(f"  Error: {e}")


def main() -> None:
    if len(sys.argv) > 1:
        try:
            seconds = parse_duration(" ".join(sys.argv[1:]))
            run_countdown(seconds)
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: python main.py 5m  (or 1h30m, 90, 1:30:00)")
        return

    print("Countdown Timer")
    print("=" * 40)

    while True:
        print("\n1. Start new timer")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break
        elif choice == "1":
            seconds, label = get_duration_from_user()
            run_countdown(seconds, label)
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
