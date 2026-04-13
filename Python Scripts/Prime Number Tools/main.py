"""Prime Number Tools — CLI tool.

Check primality, generate primes up to N (Sieve of Eratosthenes),
find prime factors, compute GCD/LCM, and find twin primes.

Usage:
    python main.py
"""

import math


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Miller-Rabin deterministic for n < 3,215,031,751."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    # Trial division up to sqrt(n)
    r = int(math.isqrt(n))
    f = 5
    while f <= r:
        if n % f == 0 or n % (f + 2) == 0:
            return False
        f += 6
    return True


def sieve(limit: int) -> list[int]:
    """Return all primes ≤ limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
    return [i for i, v in enumerate(is_p) if v]


def prime_factors(n: int) -> list[int]:
    """Return sorted list of prime factors (with repetition)."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def factorization_str(n: int) -> str:
    from collections import Counter
    factors = prime_factors(n)
    counts  = Counter(factors)
    parts   = [f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(counts.items())]
    return " × ".join(parts)


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def twin_primes(limit: int) -> list[tuple[int, int]]:
    primes = sieve(limit)
    return [(p, p + 2) for p in primes if is_prime(p + 2) and p + 2 <= limit]


def nth_prime(n: int) -> int:
    """Return the nth prime (1-indexed)."""
    if n < 1:
        return -1
    count   = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate


def goldbach(n: int) -> list[tuple[int, int]]:
    """Find all Goldbach pairs for even n > 2."""
    if n <= 2 or n % 2 != 0:
        return []
    primes = set(sieve(n))
    pairs  = [(p, n - p) for p in primes if n - p in primes and p <= n - p]
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Prime Number Tools
------------------
1. Check if number is prime
2. Generate primes up to N
3. Prime factorization
4. GCD and LCM
5. Twin primes up to N
6. Find Nth prime
7. Goldbach conjecture pairs
0. Quit
"""


def get_int(prompt: str) -> int | None:
    try:
        return int(input(prompt).strip().replace(",", ""))
    except ValueError:
        print("  Please enter an integer.")
        return None


def main() -> None:
    print("Prime Number Tools")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            n = get_int("  Number: ")
            if n is None:
                continue
            result = is_prime(n)
            if result:
                print(f"\n  ✓ {n:,} IS prime.")
            else:
                # Find nearest primes
                lo = n - 1
                hi = n + 1
                while lo > 1 and not is_prime(lo):
                    lo -= 1
                while not is_prime(hi):
                    hi += 1
                print(f"\n  ✗ {n:,} is NOT prime.")
                print(f"  Nearest primes: {lo:,} (below) and {hi:,} (above)")

        elif choice == "2":
            n = get_int("  Generate primes up to N: ")
            if n is None:
                continue
            if n > 10_000_000:
                print("  Limit too large (max 10,000,000).")
                continue
            primes = sieve(n)
            print(f"\n  Found {len(primes):,} primes ≤ {n:,}")
            if primes:
                print(f"  First 20: {primes[:20]}")
                print(f"  Last  10: {primes[-10:]}")

        elif choice == "3":
            n = get_int("  Number to factorize: ")
            if n is None or n < 2:
                continue
            factors = prime_factors(n)
            print(f"\n  {n:,} = {factorization_str(n)}")
            print(f"  Factors: {factors}")

        elif choice == "4":
            a = get_int("  First number : ")
            b = get_int("  Second number: ")
            if a is None or b is None:
                continue
            print(f"\n  GCD({a}, {b}) = {gcd(a, b):,}")
            print(f"  LCM({a}, {b}) = {lcm(a, b):,}")

        elif choice == "5":
            n = get_int("  Find twin primes up to N: ")
            if n is None:
                continue
            if n > 1_000_000:
                print("  Limit too large (max 1,000,000).")
                continue
            twins = twin_primes(n)
            print(f"\n  Found {len(twins):,} twin prime pairs ≤ {n:,}")
            for pair in twins[:20]:
                print(f"  ({pair[0]:,}, {pair[1]:,})")
            if len(twins) > 20:
                print(f"  ... and {len(twins) - 20} more")

        elif choice == "6":
            n = get_int("  N (find the Nth prime): ")
            if n is None or n < 1:
                continue
            if n > 100_000:
                print("  N too large (max 100,000).")
                continue
            p = nth_prime(n)
            print(f"\n  The {n:,}th prime is {p:,}")

        elif choice == "7":
            n = get_int("  Even number ≥ 4: ")
            if n is None or n < 4 or n % 2 != 0:
                print("  Must be an even number ≥ 4.")
                continue
            pairs = goldbach(n)
            print(f"\n  Goldbach pairs for {n:,}:")
            for a, b in pairs[:10]:
                print(f"    {a:,} + {b:,} = {n:,}")
            if len(pairs) > 10:
                print(f"  ... and {len(pairs) - 10} more pairs")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
