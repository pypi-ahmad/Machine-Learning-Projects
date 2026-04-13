"""Equation Solver — CLI tool.

Solve linear, quadratic, cubic, and systems of linear equations.
Supports symbolic-style input and step-by-step output.

Usage:
    python main.py
    python main.py --linear "2x+3=7"
    python main.py --quadratic "1 -3 2"
    python main.py --system "2 1 5" "1 -1 1"
"""

import argparse
import math
import re
import sys


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt(n: float) -> str:
    """Format number: drop .0 for integers."""
    return str(int(n)) if n == int(n) else f"{n:.6g}"


def solve_linear(a: float, b: float, c: float) -> str:
    """Solve ax + b = c  →  x = (c-b)/a."""
    if a == 0:
        return "No solution (a=0)." if b != c else "Infinite solutions (identity)."
    x = (c - b) / a
    return f"x = {fmt(x)}"


def solve_quadratic(a: float, b: float, c: float) -> str:
    """Solve ax² + bx + c = 0."""
    if a == 0:
        return solve_linear(b, c, 0)
    disc = b * b - 4 * a * c
    lines = [f"Discriminant = {fmt(disc)}"]
    if disc > 0:
        sq = math.sqrt(disc)
        x1 = (-b + sq) / (2 * a)
        x2 = (-b - sq) / (2 * a)
        lines.append(f"Two real roots:  x₁ = {fmt(x1)},  x₂ = {fmt(x2)}")
    elif disc == 0:
        x = -b / (2 * a)
        lines.append(f"One real root:   x = {fmt(x)}")
    else:
        real = -b / (2 * a)
        imag = math.sqrt(-disc) / (2 * a)
        lines.append(f"Complex roots:   x = {fmt(real)} ± {fmt(imag)}i")
    return "\n".join(lines)


def solve_cubic(a: float, b: float, c: float, d: float) -> str:
    """Solve ax³ + bx² + cx + d = 0 via Cardano / numerical Newton."""
    if a == 0:
        return solve_quadratic(b, c, d)

    def f(x):  return a*x**3 + b*x**2 + c*x + d
    def fp(x): return 3*a*x**2 + 2*b*x + c

    roots = []
    for x0 in [-100, -10, -1, 0, 1, 10, 100]:
        x = float(x0)
        for _ in range(200):
            fx = f(x)
            if abs(fx) < 1e-12: break
            dfx = fp(x)
            if dfx == 0: x += 0.1; continue
            x -= fx / dfx
        if abs(f(x)) < 1e-8:
            if not any(abs(x - r) < 1e-6 for r in roots):
                roots.append(x)

    if not roots:
        return "Could not find real roots numerically."
    return "Real roots: " + ",  ".join(f"x = {fmt(round(r, 8))}" for r in sorted(roots))


def solve_2x2(a1, b1, c1, a2, b2, c2) -> str:
    """Solve  a1x + b1y = c1  and  a2x + b2y = c2  (Cramer's rule)."""
    det = a1 * b2 - a2 * b1
    if det == 0:
        return "No unique solution (determinant = 0)."
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return f"x = {fmt(x)},  y = {fmt(y)}"


def parse_linear_eq(expr: str):
    """Parse 'ax+b=c' or 'ax=c' into (a, b, c)."""
    expr = expr.replace(" ", "").replace("–", "-")
    if "=" not in expr:
        raise ValueError("Equation must contain '='.")
    lhs, rhs = expr.split("=", 1)
    c = float(rhs)
    # coefficient of x
    m = re.fullmatch(r"([+-]?\d*\.?\d*)x([+-]\d+\.?\d*)?", lhs)
    if not m:
        raise ValueError(f"Cannot parse '{lhs}'.")
    a_str = m.group(1) or "1"
    a = float(a_str) if a_str not in ("", "+", "-") else float(a_str + "1")
    b = float(m.group(2) or 0)
    return a, b, c


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive():
    print("=== Equation Solver ===")
    print("Commands: linear | quadratic | cubic | system | quit\n")
    while True:
        cmd = input("Type: ").strip().lower()
        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "linear":
            print("Enter equation as  ax+b=c  (e.g. 3x+2=11)")
            eq = input("> ").strip()
            try:
                a, b, c = parse_linear_eq(eq)
                print(f"  {solve_linear(a, b, c)}\n")
            except Exception as e:
                print(f"  Error: {e}\n")
        elif cmd == "quadratic":
            print("Enter coefficients a b c for  ax²+bx+c=0")
            try:
                a, b, c = map(float, input("> ").split())
                print(f"  {solve_quadratic(a, b, c)}\n")
            except Exception as e:
                print(f"  Error: {e}\n")
        elif cmd == "cubic":
            print("Enter coefficients a b c d for  ax³+bx²+cx+d=0")
            try:
                a, b, c, d = map(float, input("> ").split())
                print(f"  {solve_cubic(a, b, c, d)}\n")
            except Exception as e:
                print(f"  Error: {e}\n")
        elif cmd == "system":
            print("2×2 system:  a1x+b1y=c1  and  a2x+b2y=c2")
            print("Enter row 1 (a1 b1 c1):")
            try:
                a1, b1, c1 = map(float, input("> ").split())
                print("Enter row 2 (a2 b2 c2):")
                a2, b2, c2 = map(float, input("> ").split())
                print(f"  {solve_2x2(a1, b1, c1, a2, b2, c2)}\n")
            except Exception as e:
                print(f"  Error: {e}\n")
        else:
            print("  Unknown command.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Equation Solver")
    parser.add_argument("--linear",     metavar="EQ",   help="Linear equation, e.g. '3x+2=11'")
    parser.add_argument("--quadratic",  metavar="A B C", nargs=3, type=float,
                        help="Quadratic coefficients a b c")
    parser.add_argument("--cubic",      metavar="A B C D", nargs=4, type=float,
                        help="Cubic coefficients a b c d")
    parser.add_argument("--system",     metavar="ROW",  nargs=2,
                        help="2×2 system: '2 1 5' '1 -1 1'")
    args = parser.parse_args()

    if args.linear:
        a, b, c = parse_linear_eq(args.linear)
        print(solve_linear(a, b, c))
    elif args.quadratic:
        print(solve_quadratic(*args.quadratic))
    elif args.cubic:
        print(solve_cubic(*args.cubic))
    elif args.system:
        r1 = list(map(float, args.system[0].split()))
        r2 = list(map(float, args.system[1].split()))
        print(solve_2x2(*r1, *r2))
    else:
        interactive()


if __name__ == "__main__":
    main()
