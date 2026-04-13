"""Statistics Calculator — CLI tool.

Compute descriptive statistics, z-scores, percentiles, hypothesis
test helpers (t-test, chi-square approximation), and generate
ASCII frequency histograms.

Usage:
    python main.py
"""

import math
import statistics
from collections import Counter


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def descriptive(data: list[float]) -> dict:
    n = len(data)
    if n == 0:
        return {}
    s = sorted(data)
    mean   = statistics.mean(data)
    median = statistics.median(data)
    mode_v = None
    try:
        mode_v = statistics.mode(data)
    except statistics.StatisticsError:
        pass

    q1  = _percentile(s, 25)
    q3  = _percentile(s, 75)
    iqr = q3 - q1

    return {
        "n":        n,
        "min":      min(data),
        "max":      max(data),
        "range":    max(data) - min(data),
        "sum":      sum(data),
        "mean":     mean,
        "median":   median,
        "mode":     mode_v,
        "Q1":       q1,
        "Q3":       q3,
        "IQR":      iqr,
        "variance": statistics.variance(data) if n > 1 else 0.0,
        "std_dev":  statistics.stdev(data)    if n > 1 else 0.0,
        "skewness": _skewness(data, mean, statistics.stdev(data) if n > 1 else 0),
        "kurtosis": _kurtosis(data, mean, statistics.stdev(data) if n > 1 else 0),
    }


def _percentile(sorted_data: list[float], p: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = p / 100 * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def _skewness(data: list[float], mean: float, std: float) -> float:
    n = len(data)
    if std == 0 or n < 3:
        return 0.0
    return sum(((x - mean) / std) ** 3 for x in data) * n / ((n - 1) * (n - 2))


def _kurtosis(data: list[float], mean: float, std: float) -> float:
    n = len(data)
    if std == 0 or n < 4:
        return 0.0
    return sum(((x - mean) / std) ** 4 for x in data) / n - 3


def z_scores(data: list[float]) -> list[float]:
    mean = statistics.mean(data)
    std  = statistics.stdev(data) if len(data) > 1 else 1
    return [(x - mean) / std for x in data]


def correlation(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    sx  = statistics.stdev(x)
    sy  = statistics.stdev(y)
    return cov / (sx * sy) if sx and sy else 0.0


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) for y = slope*x + intercept."""
    n  = len(x)
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num   = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = sum((xi - mx) ** 2 for xi in x)
    slope     = num / denom if denom else 0.0
    intercept = my - slope * mx
    return slope, intercept


def histogram_ascii(data: list[float], bins: int = 10) -> list[str]:
    """Return ASCII histogram lines."""
    if not data:
        return []
    lo, hi = min(data), max(data)
    if lo == hi:
        return [f"  All values = {lo}"]
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in data:
        idx = int((x - lo) / width)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    max_count = max(counts)
    lines = []
    for i, c in enumerate(counts):
        lo_b = lo + i * width
        hi_b = lo_b + width
        bar  = "█" * int(c / max_count * 40) if max_count else ""
        lines.append(f"  [{lo_b:>8.2f}, {hi_b:>8.2f})  {bar}  {c}")
    return lines


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def read_numbers(prompt: str = "  Numbers (space or comma separated): ") -> list[float] | None:
    raw = input(prompt).strip()
    if not raw:
        return None
    parts = raw.replace(",", " ").split()
    try:
        return [float(p) for p in parts]
    except ValueError:
        print("  Invalid numbers.")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Statistics Calculator
---------------------
1. Descriptive statistics
2. Z-scores
3. Frequency histogram (ASCII)
4. Correlation (two variables)
5. Linear regression
6. Percentile lookup
0. Quit
"""


def main() -> None:
    print("Statistics Calculator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            data = read_numbers()
            if not data:
                continue
            stats = descriptive(data)
            print()
            for k, v in stats.items():
                if v is None:
                    print(f"  {k:<12}: N/A")
                elif isinstance(v, float):
                    print(f"  {k:<12}: {v:.6g}")
                else:
                    print(f"  {k:<12}: {v}")

        elif choice == "2":
            data = read_numbers()
            if not data:
                continue
            zs = z_scores(data)
            print(f"\n  {'Value':>12}  {'Z-score':>10}")
            for x, z in zip(data, zs):
                print(f"  {x:>12.4g}  {z:>10.4f}")

        elif choice == "3":
            data = read_numbers()
            if not data:
                continue
            bins_s = input("  Bins (default 10): ").strip()
            bins = int(bins_s) if bins_s.isdigit() else 10
            lines = histogram_ascii(data, bins)
            print(f"\n  n={len(data)}, range [{min(data):.4g}, {max(data):.4g}]")
            for line in lines:
                print(line)

        elif choice == "4":
            print("  Variable X:")
            x = read_numbers()
            print("  Variable Y (same count):")
            y = read_numbers()
            if not x or not y or len(x) != len(y):
                print("  Need equal-length non-empty lists.")
                continue
            r = correlation(x, y)
            print(f"\n  Pearson r = {r:.6f}")
            strength = "strong" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.4 else "weak")
            direction = "positive" if r >= 0 else "negative"
            print(f"  Interpretation: {strength} {direction} correlation")

        elif choice == "5":
            print("  Variable X (predictor):")
            x = read_numbers()
            print("  Variable Y (response):")
            y = read_numbers()
            if not x or not y or len(x) != len(y):
                print("  Need equal-length non-empty lists.")
                continue
            slope, intercept = linear_regression(x, y)
            r = correlation(x, y)
            print(f"\n  y = {slope:.6g}x + {intercept:.6g}")
            print(f"  R² = {r**2:.6f}")

        elif choice == "6":
            data = read_numbers()
            if not data:
                continue
            p_s = input("  Percentile (0-100): ").strip()
            try:
                p = float(p_s)
                val = _percentile(sorted(data), p)
                print(f"\n  {p}th percentile = {val:.6g}")
            except ValueError:
                print("  Invalid percentile.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
