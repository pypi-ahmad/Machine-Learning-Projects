"""Fibonacci Generator — CLI tool.

Generate Fibonacci sequences, find the Nth term, check if a number
is Fibonacci, and explore related sequences (Lucas, Tribonacci,
Padovan).  Uses fast matrix exponentiation for large N.

Usage:
    python main.py
"""

import math


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def fibonacci_sequence(n: int) -> list[int]:
    """Return first n Fibonacci numbers (F(1)=1, F(2)=1, ...)."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    seq = [1, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq


def _mat_mul(a: list, b: list) -> list:
    """2×2 matrix multiply."""
    return [
        a[0]*b[0] + a[1]*b[2],  a[0]*b[1] + a[1]*b[3],
        a[2]*b[0] + a[3]*b[2],  a[2]*b[1] + a[3]*b[3],
    ]


def _mat_pow(m: list, p: int) -> list:
    result = [1, 0, 0, 1]  # identity
    while p:
        if p & 1:
            result = _mat_mul(result, m)
        m = _mat_mul(m, m)
        p >>= 1
    return result


def nth_fibonacci(n: int) -> int:
    """Return F(n) in O(log n) using matrix exponentiation. F(1)=F(2)=1."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    m = _mat_pow([1, 1, 1, 0], n - 1)
    return m[0]


def is_fibonacci(n: int) -> bool:
    """A number is Fibonacci iff 5n²±4 is a perfect square."""
    if n < 0:
        return False
    def is_perfect_sq(x):
        s = int(math.isqrt(x))
        return s * s == x
    return is_perfect_sq(5 * n * n + 4) or is_perfect_sq(5 * n * n - 4)


def golden_ratio_convergents(n: int) -> list[float]:
    """Return F(n+1)/F(n) for the first n terms (converges to φ)."""
    seq = fibonacci_sequence(n + 1)
    return [seq[i+1] / seq[i] for i in range(len(seq) - 1)]


def lucas_sequence(n: int) -> list[int]:
    """Lucas numbers: L(1)=1, L(2)=3, L(n)=L(n-1)+L(n-2)."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    seq = [1, 3]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def tribonacci(n: int) -> list[int]:
    """T(1)=T(2)=1, T(3)=2, T(n)=T(n-1)+T(n-2)+T(n-3)."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    seq = [1, 1, 2]
    for _ in range(3, n):
        seq.append(seq[-1] + seq[-2] + seq[-3])
    return seq[:n]


def fibonacci_properties(n: int) -> dict:
    f = nth_fibonacci(n)
    prev = nth_fibonacci(n - 1) if n > 1 else 0
    nxt  = nth_fibonacci(n + 1)
    phi  = (1 + math.sqrt(5)) / 2
    return {
        f"F({n})":   f,
        f"F({n-1})": prev,
        f"F({n+1})": nxt,
        "Ratio F(n+1)/F(n)": nxt / f if f else "N/A",
        "Golden ratio φ":    round(phi, 10),
        "Is perfect square": int(math.isqrt(f)) ** 2 == f,
        "Is even":           f % 2 == 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Fibonacci Generator
-------------------
1. Generate first N Fibonacci numbers
2. Find the Nth Fibonacci number
3. Check if number is Fibonacci
4. Fibonacci properties at N
5. Lucas sequence
6. Tribonacci sequence
7. Golden ratio convergents
0. Quit
"""


def main() -> None:
    print("Fibonacci Generator")
    print(f"  φ (Golden ratio) = {(1+math.sqrt(5))/2:.10f}")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            n_s = input("  N: ").strip()
            try:
                n = int(n_s)
                if n > 10_000:
                    print("  N too large (max 10,000).")
                    continue
                seq = fibonacci_sequence(n)
                print(f"\n  First {n} Fibonacci numbers:")
                for i in range(0, len(seq), 10):
                    print("  " + "  ".join(f"{v:>12,}" for v in seq[i:i+10]))
            except ValueError:
                print("  Invalid N.")

        elif choice == "2":
            n_s = input("  N: ").strip()
            try:
                n = int(n_s)
                if n > 100_000:
                    print("  N too large (max 100,000).")
                    continue
                f = nth_fibonacci(n)
                print(f"\n  F({n}) = {f:,}")
                print(f"  Digits: {len(str(f))}")
            except ValueError:
                print("  Invalid N.")

        elif choice == "3":
            n_s = input("  Number: ").strip().replace(",", "")
            try:
                n = int(n_s)
                if is_fibonacci(n):
                    print(f"\n  ✓ {n:,} IS a Fibonacci number.")
                else:
                    print(f"\n  ✗ {n:,} is NOT a Fibonacci number.")
            except ValueError:
                print("  Invalid number.")

        elif choice == "4":
            n_s = input("  N: ").strip()
            try:
                n = int(n_s)
                props = fibonacci_properties(n)
                print()
                for k, v in props.items():
                    print(f"  {k:<25}: {v}")
            except ValueError:
                print("  Invalid N.")

        elif choice == "5":
            n_s = input("  N: ").strip()
            try:
                n = int(n_s)
                seq = lucas_sequence(n)
                print(f"\n  Lucas sequence (first {n}):")
                print("  " + "  ".join(f"{v:,}" for v in seq[:20]))
            except ValueError:
                print("  Invalid N.")

        elif choice == "6":
            n_s = input("  N: ").strip()
            try:
                n = int(n_s)
                seq = tribonacci(n)
                print(f"\n  Tribonacci (first {n}):")
                print("  " + "  ".join(f"{v:,}" for v in seq[:20]))
            except ValueError:
                print("  Invalid N.")

        elif choice == "7":
            n_s = input("  N: ").strip()
            try:
                n = int(n_s)
                convs = golden_ratio_convergents(min(n, 50))
                phi = (1 + math.sqrt(5)) / 2
                print(f"\n  {'n':>5}  {'F(n+1)/F(n)':>18}  {'Error':>12}")
                for i, r in enumerate(convs, 1):
                    print(f"  {i:>5}  {r:>18.12f}  {abs(r-phi):>12.2e}")
            except ValueError:
                print("  Invalid N.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
