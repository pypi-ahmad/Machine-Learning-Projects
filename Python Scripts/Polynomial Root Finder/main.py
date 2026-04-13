"""Polynomial Root Finder — CLI tool.

Find all real and complex roots of polynomials of any degree.
Uses Durand-Kerner method (Weierstrass iteration) for general polynomials.

Usage:
    python main.py
    python main.py --coeffs "1 -6 11 -6"     # x³ - 6x² + 11x - 6
    python main.py --coeffs "1 0 -5 0 4"     # x⁴ - 5x² + 4
"""

import argparse
import cmath
import math
import sys


def fmt_complex(z: complex, tol: float = 1e-8) -> str:
    """Format complex number, hiding tiny imaginary parts."""
    if abs(z.imag) < tol:
        r = z.real
        return str(int(r)) if abs(r - round(r)) < tol else f"{r:.6g}"
    if abs(z.real) < tol:
        return f"{z.imag:.6g}i"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.6g} {sign} {abs(z.imag):.6g}i"


def poly_eval(coeffs: list[complex], z: complex) -> complex:
    """Horner's method: evaluate polynomial at z."""
    result = complex(0)
    for c in coeffs:
        result = result * z + c
    return result


def durand_kerner(coeffs: list[float], max_iter: int = 200, tol: float = 1e-12) -> list[complex]:
    """
    Durand-Kerner (Weierstrass) method to find all roots of a polynomial.
    coeffs: [a_n, a_{n-1}, ..., a_1, a_0] (highest degree first).
    """
    n = len(coeffs) - 1    # degree
    if n == 0:
        return []

    # Normalize: leading coefficient = 1
    lead = coeffs[0]
    c = [x / lead for x in coeffs]

    # Initial approximations on a circle of radius bounded by Cauchy's bound
    r = 1 + max(abs(ci) for ci in c[1:])
    roots = [
        complex(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) * r
        for k in range(n)
    ]

    for _ in range(max_iter):
        new_roots = []
        for i, zi in enumerate(roots):
            denom = complex(1)
            for j, zj in enumerate(roots):
                if i != j:
                    denom *= (zi - zj)
            if abs(denom) < 1e-30:
                new_roots.append(zi)
                continue
            new_roots.append(zi - poly_eval(c, zi) / denom)
        delta = max(abs(new_roots[i] - roots[i]) for i in range(n))
        roots = new_roots
        if delta < tol:
            break

    return roots


def parse_coefficients(s: str) -> list[float]:
    """Parse space or comma-separated coefficients."""
    return [float(x) for x in s.replace(",", " ").split()]


def polynomial_str(coeffs: list[float]) -> str:
    """Format polynomial as human-readable string."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        deg = n - i
        if c == 0: continue
        sign = "+" if c > 0 else "-"
        val  = abs(c)
        val_s = str(int(val)) if val == int(val) else f"{val:.6g}"
        if deg == 0:
            terms.append(f"{sign} {val_s}")
        elif deg == 1:
            terms.append(f"{sign} {val_s}x")
        else:
            terms.append(f"{sign} {val_s}x^{deg}")
    if not terms:
        return "0"
    result = " ".join(terms).lstrip("+ ").strip()
    return result


def find_and_display(coeffs: list[float]) -> None:
    if len(coeffs) < 2:
        print("  Need at least 2 coefficients (degree ≥ 1).")
        return
    # Remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs = coeffs[1:]
    n = len(coeffs) - 1

    print(f"\n  Polynomial: f(x) = {polynomial_str(coeffs)}")
    print(f"  Degree: {n}")

    roots = durand_kerner(coeffs)

    real_roots    = sorted([r for r in roots if abs(r.imag) < 1e-7], key=lambda z: z.real)
    complex_roots = [r for r in roots if abs(r.imag) >= 1e-7]

    print(f"\n  {'─'*40}")
    if real_roots:
        print(f"  Real roots ({len(real_roots)}):")
        for r in real_roots:
            print(f"    x = {fmt_complex(r)}")
    if complex_roots:
        print(f"  Complex roots ({len(complex_roots)}):")
        # Group conjugate pairs
        done = set()
        for i, r in enumerate(complex_roots):
            if i in done: continue
            print(f"    x = {fmt_complex(r)}")
            # Find conjugate
            for j, s in enumerate(complex_roots):
                if j != i and j not in done and abs(r - s.conjugate()) < 1e-6:
                    print(f"    x = {fmt_complex(s)}")
                    done.add(j); done.add(i); break
    if not real_roots and not complex_roots:
        print("  No roots found.")

    # Verify
    max_err = max(abs(poly_eval(coeffs, r)) for r in roots) if roots else 0
    print(f"\n  Max residual |f(root)|: {max_err:.2e}")


def interactive():
    print("=== Polynomial Root Finder ===")
    print("Enter coefficients highest-degree first.")
    print("Example: '1 -6 11 -6'  →  x³ - 6x² + 11x - 6\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        if line.lower() in ("quit", "q", "exit"): break
        try:
            coeffs = parse_coefficients(line)
            find_and_display(coeffs)
        except Exception as e:
            print(f"  Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Polynomial Root Finder")
    parser.add_argument("--coeffs", metavar="'a b c ...'",
                        help="Coefficients, highest degree first: '1 -3 2'")
    args = parser.parse_args()

    if args.coeffs:
        try:
            coeffs = parse_coefficients(args.coeffs)
            find_and_display(coeffs)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
