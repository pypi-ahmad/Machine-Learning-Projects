"""Fraction Calculator — CLI tool.

Perform arithmetic with fractions: add, subtract, multiply, divide.
Convert between fractions, decimals, and mixed numbers.
Find continued fraction expansions and best rational approximations.

Usage:
    python main.py
"""

import math
from fractions import Fraction


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def parse_fraction(s: str) -> Fraction:
    """Parse '3/4', '1 2/3', '0.75', '-5', etc."""
    s = s.strip()
    # Mixed number: "1 2/3"
    if " " in s:
        parts = s.split()
        whole = int(parts[0])
        frac  = Fraction(parts[1])
        return Fraction(whole) + (frac if whole >= 0 else -frac)
    return Fraction(s)


def to_mixed(f: Fraction) -> str:
    """Convert improper fraction to mixed number string."""
    if f.denominator == 1:
        return str(f.numerator)
    whole = int(f)
    rem   = abs(f) - abs(whole)
    if whole == 0:
        return str(f)
    return f"{whole} {rem.numerator}/{rem.denominator}"


def to_decimal(f: Fraction, places: int = 10) -> str:
    return f"{float(f):.{places}g}"


def is_repeating(f: Fraction) -> bool:
    """Check if the decimal representation is repeating."""
    d = f.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    return d > 1


def continued_fraction(f: Fraction, max_terms: int = 20) -> list[int]:
    """Return continued fraction coefficients [a0; a1, a2, ...]."""
    result = []
    for _ in range(max_terms):
        a = int(f)
        result.append(a)
        f = f - a
        if f == 0:
            break
        f = Fraction(1, f)
    return result


def best_rational(decimal_val: float, max_denom: int = 1000) -> Fraction:
    """Find best rational approximation using Stern-Brocot."""
    return Fraction(decimal_val).limit_denominator(max_denom)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Fraction Calculator
-------------------
1. Add two fractions
2. Subtract
3. Multiply
4. Divide
5. Power (fraction ^ integer)
6. Convert fraction ↔ decimal
7. Simplify fraction
8. Continued fraction expansion
9. Best rational approximation
0. Quit
"""


def read_fraction(prompt: str) -> Fraction | None:
    try:
        return parse_fraction(input(prompt).strip())
    except (ValueError, ZeroDivisionError) as e:
        print(f"  Invalid fraction: {e}")
        return None


def show(f: Fraction, label: str = "Result") -> None:
    print(f"\n  {label}:")
    print(f"    Fraction : {f}")
    print(f"    Mixed    : {to_mixed(f)}")
    print(f"    Decimal  : {to_decimal(f)}")
    if f.denominator != 1:
        repeating = "repeating" if is_repeating(f) else "terminating"
        print(f"    Decimal type: {repeating}")


def main() -> None:
    print("Fraction Calculator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice in ("1", "2", "3", "4"):
            a = read_fraction("  First fraction  : ")
            b = read_fraction("  Second fraction : ")
            if a is None or b is None:
                continue
            if choice == "1":
                show(a + b, f"{a} + {b}")
            elif choice == "2":
                show(a - b, f"{a} - {b}")
            elif choice == "3":
                show(a * b, f"{a} × {b}")
            else:
                if b == 0:
                    print("  Division by zero.")
                    continue
                show(a / b, f"{a} ÷ {b}")

        elif choice == "5":
            a = read_fraction("  Fraction: ")
            if a is None:
                continue
            exp_s = input("  Exponent (integer): ").strip()
            try:
                exp = int(exp_s)
                show(a ** exp, f"{a}^{exp}")
            except ValueError:
                print("  Invalid exponent.")

        elif choice == "6":
            sub = input("  (f)raction to decimal or (d)ecimal to fraction? ").strip().lower()
            if sub.startswith("f"):
                a = read_fraction("  Fraction: ")
                if a is not None:
                    print(f"\n  Decimal: {float(a):.15g}")
            else:
                d_s = input("  Decimal: ").strip()
                try:
                    d = float(d_s)
                    f = Fraction(d).limit_denominator(10_000)
                    show(f, f"≈ {d}")
                except ValueError:
                    print("  Invalid decimal.")

        elif choice == "7":
            a = read_fraction("  Fraction: ")
            if a is not None:
                show(a, f"Simplified form of {a}")

        elif choice == "8":
            a = read_fraction("  Fraction: ")
            if a is None:
                continue
            cf = continued_fraction(a)
            print(f"\n  [{cf[0]}; {', '.join(str(c) for c in cf[1:])}]")
            # Show convergents
            print(f"  Convergents:")
            h_prev, h_curr = 1, cf[0]
            k_prev, k_curr = 0, 1
            for i, ai in enumerate(cf[1:], 1):
                h_prev, h_curr = h_curr, ai * h_curr + h_prev
                k_prev, k_curr = k_curr, ai * k_curr + k_prev
                print(f"    [{i}] {h_curr}/{k_curr} = {h_curr/k_curr:.8f}")

        elif choice == "9":
            d_s = input("  Decimal (e.g. 3.14159): ").strip()
            max_d_s = input("  Max denominator (default 1000): ").strip() or "1000"
            try:
                d = float(d_s)
                max_d = int(max_d_s)
                f = best_rational(d, max_d)
                print(f"\n  Best rational approx: {f}")
                print(f"  Error: {abs(d - float(f)):.2e}")
            except ValueError:
                print("  Invalid input.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
