"""Function Plotter — CLI tool.

Plot mathematical functions as ASCII graphs in the terminal.
Supports standard math functions, multiple curves, and custom ranges.

Usage:
    python main.py
    python main.py --func "sin(x)" --xmin -6.28 --xmax 6.28
    python main.py --func "x**2 - 4*x + 3" --xmin -1 --xmax 5
"""

import argparse
import math
import sys


# Safe math namespace for eval
SAFE_NS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE_NS.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})


def evaluate(expr: str, x: float) -> float:
    SAFE_NS["x"] = x
    return float(eval(expr, {"__builtins__": {}}, SAFE_NS))


def plot(expressions: list[str], x_min: float, x_max: float,
         width: int = 70, height: int = 25, symbols: str = "*#@+") -> None:
    """Render ASCII plot of one or more expressions over [x_min, x_max]."""
    xs = [x_min + i / (width - 1) * (x_max - x_min) for i in range(width)]

    # Evaluate all functions
    all_series = []
    for expr in expressions:
        ys = []
        for x in xs:
            try:
                ys.append(evaluate(expr, x))
            except Exception:
                ys.append(None)
        all_series.append(ys)

    # Compute y range from valid values
    valid_ys = [y for ys in all_series for y in ys if y is not None and math.isfinite(y)]
    if not valid_ys:
        print("  No valid values to plot.")
        return
    y_min, y_max = min(valid_ys), max(valid_ys)
    y_range = y_max - y_min or 1

    grid = [[" "] * width for _ in range(height)]

    # Draw axes
    zero_col = int(-x_min / (x_max - x_min) * (width - 1)) if x_min <= 0 <= x_max else None
    zero_row = int((y_max - 0) / y_range * (height - 1))    if y_min <= 0 <= y_max else None
    if zero_row is not None:
        for col in range(width): grid[zero_row][col] = "─"
    if zero_col is not None:
        for row in range(height): grid[row][zero_col] = "│"
    if zero_row is not None and zero_col is not None:
        grid[zero_row][zero_col] = "┼"

    # Plot curves
    for k, (expr, ys) in enumerate(zip(expressions, all_series)):
        sym = symbols[k % len(symbols)]
        for col, y in enumerate(ys):
            if y is None or not math.isfinite(y): continue
            row = int((y_max - y) / y_range * (height - 1))
            if 0 <= row < height:
                grid[row][col] = sym

    # Print
    print(f"\n  y_max = {y_max:.4g}")
    for row_idx, row in enumerate(grid):
        prefix = "  |"
        print(prefix + "".join(row))
    print(f"  └{'─'*width}")
    mid = width // 2
    x_mid = x_min + (x_max - x_min) / 2
    print(f"  {x_min:<10.4g}{x_mid:^{mid}.4g}{x_max:>10.4g}")
    print(f"  y_min = {y_min:.4g}")

    # Legend
    for k, expr in enumerate(expressions):
        sym = symbols[k % len(symbols)]
        print(f"  {sym}  {expr}")


def interactive():
    print("=== Function Plotter ===")
    print("Commands: plot | add | clear | quit")
    print("Example:  plot sin(x) from -6.28 to 6.28\n")
    exprs: list[str] = []

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        parts = line.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "q", "exit"):
            break

        elif cmd == "clear":
            exprs.clear()
            print("  Cleared all functions.")

        elif cmd == "add":
            expr = " ".join(parts[1:])
            if not expr:
                expr = input("  f(x) = ").strip()
            exprs.append(expr)
            print(f"  Added: {expr}")

        elif cmd == "plot":
            # plot [expr] [from xmin] [to xmax]
            rest    = " ".join(parts[1:])
            x_min   = -10.0
            x_max   =  10.0
            cur_expr = ""
            # Try to parse "expr from A to B"
            import re
            m = re.match(r"(.*?)\s*from\s*([-\d.eE+]+)\s*to\s*([-\d.eE+]+)", rest, re.I)
            if m:
                cur_expr = m.group(1).strip()
                x_min    = float(m.group(2))
                x_max    = float(m.group(3))
            elif rest:
                cur_expr = rest

            to_plot = exprs[:]
            if cur_expr:
                to_plot.append(cur_expr)
            if not to_plot:
                print("  No functions to plot. Add one or provide it inline.")
                continue

            try:
                plot(to_plot, x_min, x_max)
            except Exception as e:
                print(f"  Error: {e}")

        elif cmd == "list":
            if exprs:
                for i, e in enumerate(exprs): print(f"  [{i}] {e}")
            else:
                print("  No functions added.")

        else:
            print("  Commands: plot [f(x)] [from A to B] | add f(x) | clear | list | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="ASCII Function Plotter")
    parser.add_argument("--func",  action="append", dest="funcs", metavar="EXPR",
                        help="Function expression (repeatable for multiple curves)")
    parser.add_argument("--xmin",  type=float, default=-10.0, help="X range minimum")
    parser.add_argument("--xmax",  type=float, default=10.0,  help="X range maximum")
    parser.add_argument("--width", type=int,   default=70,    help="Plot width (chars)")
    parser.add_argument("--height",type=int,   default=25,    help="Plot height (lines)")
    args = parser.parse_args()

    if args.funcs:
        try:
            plot(args.funcs, args.xmin, args.xmax, args.width, args.height)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
