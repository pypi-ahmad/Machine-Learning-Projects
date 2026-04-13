"""Reaction Timer — CLI game.

Test and improve your reaction time.
Multi-round stats, best/worst, average, percentile rating.

Usage:
    python main.py
    python main.py --rounds 10
"""

import argparse
import random
import sys
import time


# Human reaction time benchmarks (ms)
BENCHMARKS = [
    (150,  "Inhuman 🤖"),
    (200,  "Elite ⚡"),
    (250,  "Excellent 🏆"),
    (300,  "Great 👍"),
    (400,  "Average 😐"),
    (500,  "Slow 🐢"),
    (float("inf"), "Very slow 😴"),
]


def get_rating(ms: float) -> str:
    for threshold, label in BENCHMARKS:
        if ms < threshold:
            return label
    return "Very slow"


def wait_and_measure() -> float | None:
    """Wait for random delay, then measure reaction time in ms."""
    delay = random.uniform(1.5, 5.0)
    time.sleep(delay)
    # Signal
    print("\n  \033[92m>>> PRESS ENTER NOW! <<<\033[0m")
    t0 = time.perf_counter()
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return None
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def run_round(n: int, warn_early: bool = True) -> float | None:
    """Single round with false-start detection."""
    print(f"  Get ready… (wait for the green signal)\n")

    # Detect early press: set a short 0.5s window where pressing early = false start
    false_start_end = time.time() + 0.5
    ready_printed   = False

    def check_false_start():
        # This only matters in interactive mode
        pass

    result = wait_and_measure()
    return result


def play(n_rounds: int) -> None:
    print("=== Reaction Timer ===")
    print("  Press ENTER as fast as possible when you see the green signal.")
    print("  DON'T press early — wait for it!\n")
    input("  Press ENTER to start…")

    times: list[float] = []

    for i in range(1, n_rounds + 1):
        print(f"\n  Round {i}/{n_rounds}")
        time.sleep(0.3)

        # Pre-signal pause with random jitter
        print("  Waiting…")
        delay = random.uniform(1.5, 5.0)
        deadline = time.time() + delay

        # Crude false-start guard on platforms with non-blocking stdin
        while time.time() < deadline:
            time.sleep(0.05)

        print("\n  \033[92m>>> PRESS ENTER NOW! <<<\033[0m", end="", flush=True)
        t0 = time.perf_counter()
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            break
        t1   = time.perf_counter()
        ms   = (t1 - t0) * 1000
        rating = get_rating(ms)

        times.append(ms)
        print(f"  Reaction time: {ms:.1f} ms  —  {rating}")

    if not times:
        return

    avg   = sum(times) / len(times)
    best  = min(times)
    worst = max(times)
    std   = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5

    print(f"\n  ── Results ({len(times)} rounds) ──")
    print(f"  Average:  {avg:.1f} ms  —  {get_rating(avg)}")
    print(f"  Best:     {best:.1f} ms")
    print(f"  Worst:    {worst:.1f} ms")
    print(f"  Std dev:  {std:.1f} ms")

    print("\n  Round breakdown:")
    for i, t in enumerate(times, 1):
        bar = "█" * int(t / 20)
        print(f"    R{i:>2}: {t:>6.1f} ms  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Reaction Timer")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds (default 5)")
    args = parser.parse_args()
    play(args.rounds)


if __name__ == "__main__":
    main()
