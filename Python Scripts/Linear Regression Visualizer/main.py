"""Linear Regression Visualizer — CLI tool.

Fit simple or multiple linear regression to data.
Compute OLS coefficients, R², residuals, and predictions.
Optional ASCII scatter plot.

Usage:
    python main.py
    python main.py --file data.csv --x col1 --y col2
    python main.py --demo
"""

import argparse
import csv
import math
import sys
from pathlib import Path


# ── Statistics helpers ────────────────────────────────────────────────────────

def mean(xs): return sum(xs) / len(xs)

def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)

def covariance(xs, ys):
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)

def pearson_r(xs, ys):
    cov = covariance(xs, ys)
    sx  = math.sqrt(variance(xs))
    sy  = math.sqrt(variance(ys))
    return cov / (sx * sy) if sx and sy else 0.0


# ── Simple linear regression ──────────────────────────────────────────────────

def simple_ols(xs, ys):
    """Return (slope, intercept) for y = slope*x + intercept."""
    m_x, m_y = mean(xs), mean(ys)
    num   = sum((x - m_x) * (y - m_y) for x, y in zip(xs, ys))
    denom = sum((x - m_x) ** 2 for x in xs)
    slope     = num / denom if denom else 0
    intercept = m_y - slope * m_x
    return slope, intercept


def r_squared(ys, y_pred):
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred))
    ss_tot = sum((y - mean(ys)) ** 2 for y in ys)
    return 1 - ss_res / ss_tot if ss_tot else 1.0


def rmse(ys, y_pred):
    return math.sqrt(sum((y - yp) ** 2 for y, yp in zip(ys, y_pred)) / len(ys))


# ── ASCII scatter plot ────────────────────────────────────────────────────────

def ascii_scatter(xs, ys, slope, intercept, width=60, height=20):
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max_x - min_x or 1
    ry = max_y - min_y or 1

    grid = [[" "] * width for _ in range(height)]

    # Plot regression line
    for col in range(width):
        x_val = min_x + col / (width - 1) * rx
        y_val = slope * x_val + intercept
        row   = int((max_y - y_val) / ry * (height - 1))
        if 0 <= row < height:
            grid[row][col] = "-"

    # Plot data points
    for x, y in zip(xs, ys):
        col = int((x - min_x) / rx * (width  - 1))
        row = int((max_y - y) / ry * (height - 1))
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "●"

    # Axes labels
    print(f"  {max_y:.2f}")
    for row in grid:
        print("  |" + "".join(row))
    print(f"  └{'─'*width}")
    print(f"  {min_x:.2f}{' '*(width-12)}{max_x:.2f}")


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(xs, ys, x_label="x", y_label="y", plot=True):
    slope, intercept = simple_ols(xs, ys)
    y_pred = [slope * x + intercept for x in xs]
    r2     = r_squared(ys, y_pred)
    rmse_v = rmse(ys, y_pred)
    r      = pearson_r(xs, ys)

    print(f"\n  Simple Linear Regression: {y_label} = f({x_label})")
    print(f"  {'─'*40}")
    print(f"  Equation:   ŷ = {slope:.6f}·x + ({intercept:.6f})")
    print(f"  R²:         {r2:.6f}")
    print(f"  RMSE:       {rmse_v:.6f}")
    print(f"  Pearson r:  {r:.6f}")
    print(f"  N:          {len(xs)}")

    if plot and len(xs) >= 2:
        print()
        ascii_scatter(xs, ys, slope, intercept)

    # Residuals summary
    residuals = [y - yp for y, yp in zip(ys, y_pred)]
    print(f"\n  Residuals: min={min(residuals):.4f}  max={max(residuals):.4f}"
          f"  mean={mean(residuals):.4f}  std={math.sqrt(variance(residuals)):.4f}")

    return slope, intercept


def predict_mode(xs, ys):
    slope, intercept = simple_ols(xs, ys)
    print(f"\n  Model: ŷ = {slope:.6f}·x + ({intercept:.6f})")
    while True:
        try:
            val = input("  Enter x value to predict (or 'done'): ").strip()
            if val.lower() in ("done", "q", ""): break
            x = float(val)
            y = slope * x + intercept
            print(f"    x = {x}  →  ŷ = {y:.6f}")
        except ValueError:
            print("  Invalid number.")


def demo_mode():
    import random
    random.seed(42)
    n  = 50
    xs = [i + random.gauss(0, 0.5) for i in range(n)]
    ys = [2.5 * x + 10 + random.gauss(0, 3) for x in xs]
    print("  Demo: y = 2.5x + 10 + noise")
    analyse(xs, ys, "x", "y")
    print(f"\n  True slope=2.5, true intercept=10")


def load_csv(path: str, x_col: str, y_col: str):
    rows = list(csv.DictReader(Path(path).open()))
    xs = [float(r[x_col]) for r in rows]
    ys = [float(r[y_col]) for r in rows]
    return xs, ys


def interactive():
    print("=== Linear Regression Visualizer ===")
    print("Commands: demo | manual | csv | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "demo":
            demo_mode()
        elif cmd == "manual":
            print("  Enter x values (space-separated):")
            xs = list(map(float, input("  x: ").split()))
            print("  Enter y values (space-separated):")
            ys = list(map(float, input("  y: ").split()))
            if len(xs) != len(ys) or len(xs) < 2:
                print("  Need equal number of x and y values (min 2)."); continue
            slope, intercept = analyse(xs, ys)
            if input("\n  Predict new values? (y/n): ").strip().lower() == "y":
                predict_mode(xs, ys)
        elif cmd == "csv":
            path   = input("  CSV file path: ").strip()
            x_col  = input("  X column name: ").strip()
            y_col  = input("  Y column name: ").strip()
            try:
                xs, ys = load_csv(path, x_col, y_col)
                analyse(xs, ys, x_col, y_col)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  Commands: demo | manual | csv | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Linear Regression Visualizer")
    parser.add_argument("--demo",  action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--file",  metavar="CSV",   help="CSV file path")
    parser.add_argument("--x",     metavar="COL",   help="X column name")
    parser.add_argument("--y",     metavar="COL",   help="Y column name")
    parser.add_argument("--noplot", action="store_true", help="Skip ASCII scatter plot")
    args = parser.parse_args()

    if args.demo:
        demo_mode()
    elif args.file:
        if not args.x or not args.y:
            parser.error("--x and --y are required with --file")
        try:
            xs, ys = load_csv(args.file, args.x, args.y)
            analyse(xs, ys, args.x, args.y, plot=not args.noplot)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr); sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
