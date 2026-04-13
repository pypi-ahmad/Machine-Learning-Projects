"""Quadratic Equation Solver — CLI tool.

Solves ax² + bx + c = 0 and higher-degree polynomials.
Shows discriminant analysis, vertex, axis of symmetry,
x-intercepts, and an ASCII parabola plot.

Usage:
    python main.py
"""

import cmath
import math


# ---------------------------------------------------------------------------
# Core solvers
# ---------------------------------------------------------------------------

def solve_quadratic(a: float, b: float, c: float) -> dict:
    """Solve ax² + bx + c = 0. Returns analysis dict."""
    if a == 0:
        if b == 0:
            return {"type": "degenerate", "c": c}
        return {"type": "linear", "root": -c / b}

    disc = b ** 2 - 4 * a * c
    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x ** 2 + b * vertex_x + c

    if disc > 0:
        r1 = (-b + math.sqrt(disc)) / (2 * a)
        r2 = (-b - math.sqrt(disc)) / (2 * a)
        roots = [r1, r2]
        root_type = "two real roots"
    elif disc == 0:
        r = -b / (2 * a)
        roots = [r]
        root_type = "one real root (double)"
    else:
        r1 = (-b + cmath.sqrt(disc)) / (2 * a)
        r2 = (-b - cmath.sqrt(disc)) / (2 * a)
        roots = [r1, r2]
        root_type = "two complex roots"

    return {
        "type":       root_type,
        "a": a, "b": b, "c": c,
        "discriminant": disc,
        "roots":      roots,
        "vertex":     (vertex_x, vertex_y),
        "axis":       vertex_x,
        "opens":      "upward" if a > 0 else "downward",
        "y_intercept": c,
    }


def evaluate_poly(coeffs: list[float], x: float) -> float:
    """Evaluate polynomial with coefficients [a_n, ..., a_1, a_0]."""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def solve_cubic(a: float, b: float, c: float, d: float) -> list[complex]:
    """Cardano's formula for ax³ + bx² + cx + d = 0."""
    # Normalize to x³ + px + q
    b /= a; c /= a; d /= a
    p = c - b ** 2 / 3
    q = 2 * b ** 3 / 27 - b * c / 3 + d
    D = (q / 2) ** 2 + (p / 3) ** 3

    def cbrt(z):
        return z ** (1 / 3) if z.real >= 0 else -((-z) ** (1 / 3))

    shift = -b / 3
    D_c = complex(D)
    u = cbrt(complex(-q / 2 + cmath.sqrt(D_c)))
    v = cbrt(complex(-q / 2 - cmath.sqrt(D_c)))
    w = complex(-0.5, math.sqrt(3) / 2)
    roots = [u + v + shift,
             w * u + w.conjugate() * v + shift,
             w.conjugate() * u + w * v + shift]
    return roots


# ---------------------------------------------------------------------------
# ASCII parabola
# ---------------------------------------------------------------------------

def ascii_parabola(a: float, b: float, c: float, width: int = 60, height: int = 15) -> list[str]:
    """Draw an ASCII parabola."""
    vx = -b / (2 * a)
    vy = c - b ** 2 / (4 * a)

    x_range = width / 4
    y_range = height / 2

    lines = []
    for row in range(height):
        y = vy + y_range - row * (2 * y_range / height)
        line = []
        for col in range(width):
            x = vx - x_range + col * (2 * x_range / width)
            curve_y = a * x ** 2 + b * x + c
            if abs(curve_y - y) < (2 * y_range / height):
                line.append("*")
            elif abs(y) < 0.1 * (2 * y_range / height):  # x-axis
                line.append("-")
            elif abs(x - vx) < (2 * x_range / width):    # axis of symmetry
                line.append("|")
            else:
                line.append(" ")
        lines.append("  " + "".join(line))
    return lines


def _fmt(v) -> str:
    if isinstance(v, complex):
        if abs(v.imag) < 1e-10:
            return f"{v.real:.6g}"
        return f"{v.real:.4g} + {v.imag:.4g}i"
    return f"{v:.6g}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Quadratic Equation Solver
-------------------------
1. Solve quadratic (ax² + bx + c = 0)
2. Solve linear   (ax + b = 0)
3. Solve cubic    (ax³ + bx² + cx + d = 0)
4. Plot parabola  (ASCII)
5. Evaluate polynomial
0. Quit
"""


def get_float(prompt: str) -> float | None:
    try:
        return float(input(prompt).strip())
    except ValueError:
        print("  Invalid number.")
        return None


def main() -> None:
    print("Quadratic Equation Solver")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            a = get_float("  a (coefficient of x²): ")
            b = get_float("  b (coefficient of x ): ")
            c = get_float("  c (constant)          : ")
            if None in (a, b, c):
                continue
            res = solve_quadratic(a, b, c)
            print(f"\n  Equation : {a}x² + {b}x + {c} = 0")
            print(f"  Type     : {res['type']}")
            print(f"  Disc.    : {res.get('discriminant', 'N/A')}")
            print(f"  Roots    : {', '.join(_fmt(r) for r in res.get('roots', []))}")
            vx, vy = res.get('vertex', ('N/A', 'N/A'))
            print(f"  Vertex   : ({_fmt(vx)}, {_fmt(vy)})")
            print(f"  Axis     : x = {_fmt(res.get('axis', 'N/A'))}")
            print(f"  Opens    : {res.get('opens', 'N/A')}")
            print(f"  Y-intcpt : {_fmt(res.get('y_intercept', 'N/A'))}")

        elif choice == "2":
            a = get_float("  a: ")
            b = get_float("  b: ")
            if None in (a, b):
                continue
            if a == 0:
                print(f"  No solution." if b != 0 else "  Infinite solutions.")
            else:
                print(f"\n  x = {_fmt(-b/a)}")

        elif choice == "3":
            a = get_float("  a: ")
            b = get_float("  b: ")
            c = get_float("  c: ")
            d = get_float("  d: ")
            if None in (a, b, c, d):
                continue
            if a == 0:
                res = solve_quadratic(b, c, d)
                roots = res.get("roots", [])
            else:
                roots = solve_cubic(a, b, c, d)
            print(f"\n  Roots: {', '.join(_fmt(r) for r in roots)}")

        elif choice == "4":
            a = get_float("  a: ")
            b = get_float("  b: ")
            c = get_float("  c: ")
            if None in (a, b, c):
                continue
            if a == 0:
                print("  Need a ≠ 0 for a parabola.")
                continue
            print()
            for line in ascii_parabola(a, b, c):
                print(line)

        elif choice == "5":
            raw = input("  Coefficients [a_n ... a_0] space separated: ").strip()
            try:
                coeffs = [float(v) for v in raw.split()]
            except ValueError:
                print("  Invalid coefficients.")
                continue
            x = get_float("  x = ")
            if x is None:
                continue
            result = evaluate_poly(coeffs, x)
            print(f"\n  P({x}) = {_fmt(result)}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
