"""Coin Toss Simulator — CLI tool.

Simulate fair and weighted coin tosses.
Track streaks, run tests, and visualize distributions.

Usage:
    python main.py
    python main.py --flips 100
    python main.py --flips 1000 --bias 0.6
"""

import argparse
import random
import sys
from collections import Counter


def flip(bias: float = 0.5) -> str:
    return "H" if random.random() < bias else "T"


def run_simulation(n: int, bias: float = 0.5) -> dict:
    results = [flip(bias) for _ in range(n)]
    counts  = Counter(results)
    h, t    = counts["H"], counts["T"]

    # Longest streak
    max_streak = cur_streak = 1
    max_char   = results[0]
    for i in range(1, len(results)):
        if results[i] == results[i - 1]:
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
                max_char   = results[i]
        else:
            cur_streak = 1

    # Runs (alternation count)
    runs = sum(1 for i in range(1, len(results)) if results[i] != results[i - 1]) + 1

    return {
        "results":    results,
        "heads":      h,
        "tails":      t,
        "total":      n,
        "bias":       bias,
        "max_streak": max_streak,
        "max_char":   max_char,
        "runs":       runs,
    }


def display(d: dict, show_seq: bool = True) -> None:
    n    = d["total"]
    h, t = d["heads"], d["tails"]
    print(f"\n{'─'*40}")
    print(f"  Flips:  {n:,}  |  Bias: {d['bias']:.1%}")
    print(f"  Heads:  {h:,} ({h/n:.2%})")
    print(f"  Tails:  {t:,} ({t/n:.2%})")
    print(f"  Longest streak: {d['max_streak']} × {d['max_char']}")
    print(f"  Number of runs: {d['runs']}")

    bar_h = "█" * int(h / n * 40)
    bar_t = "█" * int(t / n * 40)
    print(f"\n  H  {bar_h} {h/n:.1%}")
    print(f"  T  {bar_t} {t/n:.1%}")

    if show_seq and n <= 100:
        seq = "".join(d["results"])
        print(f"\n  Sequence: {seq}")


def streak_analysis(n: int, bias: float = 0.5, trials: int = 1000) -> None:
    """How often does a streak of length k appear in n flips?"""
    print(f"\n  Streak analysis: {n} flips × {trials:,} trials (bias={bias:.2f})")
    streak_counts: Counter = Counter()
    for _ in range(trials):
        results = [flip(bias) for _ in range(n)]
        cur = 1
        for i in range(1, len(results)):
            if results[i] == results[i - 1]:
                cur += 1
            else:
                streak_counts[cur] += 1
                cur = 1
        streak_counts[cur] += 1

    total_streaks = sum(streak_counts.values())
    print(f"  {'Streak Len':>12}  {'Count':>8}  {'Freq':>8}")
    for k in sorted(streak_counts):
        p = streak_counts[k] / total_streaks
        print(f"  {k:>12}  {streak_counts[k]:>8,}  {p:>8.4f}")


def interactive():
    print("=== Coin Toss Simulator ===")
    print("Commands: flip [n] [bias] | streak [n] [bias] | quit\n")
    while True:
        try:
            line = input("> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        cmd = line[0].lower()

        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "flip":
            n    = int(line[1]) if len(line) > 1 else 10
            bias = float(line[2]) if len(line) > 2 else 0.5
            if not (0 < bias < 1):
                print("  Bias must be between 0 and 1."); continue
            display(run_simulation(n, bias), show_seq=True)
        elif cmd == "streak":
            n    = int(line[1]) if len(line) > 1 else 50
            bias = float(line[2]) if len(line) > 2 else 0.5
            streak_analysis(n, bias)
        else:
            print("  Commands: flip [n] [bias]  |  streak [n] [bias]  |  quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Coin Toss Simulator")
    parser.add_argument("--flips",  type=int,   default=None, metavar="N",
                        help="Number of flips")
    parser.add_argument("--bias",   type=float, default=0.5,  metavar="P",
                        help="Probability of heads (0–1, default 0.5)")
    parser.add_argument("--streak", action="store_true",
                        help="Show streak frequency analysis")
    args = parser.parse_args()

    if args.flips:
        if not (0 < args.bias < 1):
            print("Error: bias must be between 0 and 1.", file=sys.stderr)
            sys.exit(1)
        if args.streak:
            streak_analysis(args.flips, args.bias)
        else:
            display(run_simulation(args.flips, args.bias))
    else:
        interactive()


if __name__ == "__main__":
    main()
