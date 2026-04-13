"""Monte Carlo Pi Estimator — CLI tool.

Estimate π using the Monte Carlo method (random points in a unit square).
Shows convergence, error, and optional ASCII visualization.

Usage:
    python main.py
    python main.py --points 1000000
    python main.py --points 100000 --plot
"""

import argparse
import math
import random
import sys
import time


def estimate_pi(n_points: int, seed: int = None) -> dict:
    """Run Monte Carlo π estimation with n_points samples."""
    rng = random.Random(seed)
    inside   = 0
    checkpoints = []

    for i in range(1, n_points + 1):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            inside += 1
        if i in {100, 1000, 10000, 100000, 1000000} or i == n_points:
            est = 4 * inside / i
            checkpoints.append((i, est, abs(est - math.pi)))

    pi_est  = 4 * inside / n_points
    error   = abs(pi_est - math.pi)
    rel_err = error / math.pi * 100

    return {
        "n":           n_points,
        "inside":      inside,
        "pi_est":      pi_est,
        "error":       error,
        "rel_error":   rel_err,
        "checkpoints": checkpoints,
    }


def ascii_plot(n_points: int = 2000, width: int = 60, height: int = 30) -> None:
    """Draw an ASCII visualization of the Monte Carlo sampling."""
    grid = [[" "] * width for _ in range(height)]

    inside_pts  = []
    outside_pts = []

    for _ in range(n_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        col = int((x + 1) / 2 * (width  - 1))
        row = int((1 - (y + 1) / 2) * (height - 1))
        if x * x + y * y <= 1.0:
            inside_pts.append((row, col))
            grid[row][col] = "·"
        else:
            outside_pts.append((row, col))
            grid[row][col] = "+"

    # Draw circle outline (approx)
    for deg in range(0, 360, 2):
        cx = math.cos(math.radians(deg))
        cy = math.sin(math.radians(deg))
        col = int((cx + 1) / 2 * (width  - 1))
        row = int((1 - (cy + 1) / 2) * (height - 1))
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "o"

    pi_est = 4 * len(inside_pts) / n_points
    print(f"\n  Monte Carlo π estimate from {n_points:,} points: {pi_est:.5f}")
    print(f"  Actual π:  {math.pi:.5f}")
    print(f"  · = inside circle ({len(inside_pts):,})   + = outside ({len(outside_pts):,})")
    print()
    for row in grid:
        print("  " + "".join(row))


def convergence_table(checkpoints: list) -> None:
    print(f"\n  {'Points':>10}  {'π estimate':>12}  {'Error':>12}  {'Rel. Error':>12}")
    print("  " + "─" * 52)
    for n, est, err in checkpoints:
        rel = err / math.pi * 100
        print(f"  {n:>10,}  {est:>12.8f}  {err:>12.8f}  {rel:>11.4f}%")


def interactive():
    print("=== Monte Carlo π Estimator ===")
    print("Commands: run [points] | plot [points] | quit\n")
    while True:
        try:
            line = input("> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        cmd = line[0].lower()

        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "run":
            n = int(line[1]) if len(line) > 1 else 100_000
            print(f"\n  Running {n:,} points...")
            t0  = time.perf_counter()
            res = estimate_pi(n)
            dt  = time.perf_counter() - t0
            print(f"  π ≈ {res['pi_est']:.8f}")
            print(f"  Actual: {math.pi:.8f}")
            print(f"  Error:  {res['error']:.2e}  ({res['rel_error']:.4f}%)")
            print(f"  Time:   {dt:.3f}s")
            convergence_table(res["checkpoints"])
        elif cmd == "plot":
            n = int(line[1]) if len(line) > 1 else 2000
            ascii_plot(n)
        else:
            print("  Commands: run [n]  |  plot [n]  |  quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Pi Estimator")
    parser.add_argument("--points", type=int, default=None, metavar="N",
                        help="Number of random points")
    parser.add_argument("--plot",   action="store_true",
                        help="Show ASCII visualization")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.points:
        random.seed(args.seed)
        if args.plot:
            ascii_plot(min(args.points, 5000))
        t0  = time.perf_counter()
        res = estimate_pi(args.points, args.seed)
        dt  = time.perf_counter() - t0
        print(f"\n  Points:    {res['n']:,}")
        print(f"  π ≈        {res['pi_est']:.8f}")
        print(f"  Actual π:  {math.pi:.8f}")
        print(f"  Error:     {res['error']:.2e}  ({res['rel_error']:.4f}%)")
        print(f"  Time:      {dt:.3f}s")
        convergence_table(res["checkpoints"])
    else:
        interactive()


if __name__ == "__main__":
    main()
