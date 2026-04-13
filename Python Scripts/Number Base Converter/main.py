"""Number Base Converter — CLI tool.

Convert integers between binary, octal, decimal, hexadecimal,
and any custom base 2–36.  Shows conversion steps and ASCII table.

Usage:
    python main.py
"""

import string


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

DIGITS = string.digits + string.ascii_uppercase


def to_base(n: int, base: int) -> str:
    """Convert non-negative integer to given base string."""
    if base < 2 or base > 36:
        raise ValueError("Base must be 2–36.")
    if n == 0:
        return "0"
    negative = n < 0
    n = abs(n)
    result = []
    while n:
        result.append(DIGITS[n % base])
        n //= base
    s = "".join(reversed(result))
    return ("-" if negative else "") + s


def from_base(s: str, base: int) -> int:
    """Parse a string in the given base to an integer."""
    return int(s, base)


def convert_all(n: int) -> dict[str, str]:
    return {
        "Binary (2)":       to_base(n, 2),
        "Octal (8)":        to_base(n, 8),
        "Decimal (10)":     str(n),
        "Hexadecimal (16)": to_base(n, 16),
        "Base 32":          to_base(n, 32),
        "Base 36":          to_base(n, 36),
    }


def conversion_steps(n: int, base: int) -> list[str]:
    """Show repeated-division steps."""
    steps = []
    num = abs(n)
    while num > 0:
        q, r = divmod(num, base)
        steps.append(f"  {num} ÷ {base} = {q}  remainder {r}  ({DIGITS[r]})")
        num = q
    return steps


def ascii_table(start: int = 32, end: int = 127) -> list[str]:
    lines = []
    lines.append(f"  {'Dec':>4}  {'Hex':>4}  {'Oct':>4}  {'Bin':>8}  Char")
    lines.append(f"  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*4}")
    for i in range(start, end):
        ch = chr(i) if 32 <= i < 127 else " "
        lines.append(
            f"  {i:>4}  {i:>4X}  {i:>4o}  {i:>08b}  {ch!r}"
        )
    return lines


def twos_complement(n: int, bits: int = 8) -> str:
    if n >= 0:
        return bin(n)[2:].zfill(bits)
    return bin((1 << bits) + n)[2:]


def ieee754(f: float) -> dict:
    """Decompose a 32-bit float into sign, exponent, mantissa."""
    import struct
    packed  = struct.pack("!f", f)
    bits    = int.from_bytes(packed, "big")
    sign    = (bits >> 31) & 1
    exp     = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return {
        "binary":   f"{bits:032b}",
        "sign":     sign,
        "exponent": exp,
        "exp_actual": exp - 127,
        "mantissa": f"{mantissa:023b}",
        "hex":      f"{bits:08X}",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Number Base Converter
---------------------
1. Convert decimal to all common bases
2. Convert to custom base
3. Convert from any base to decimal
4. Show conversion steps
5. Two's complement
6. ASCII table (Dec/Hex/Oct/Bin)
7. IEEE 754 float decomposition
0. Quit
"""


def get_int(prompt: str) -> int | None:
    try:
        val = input(prompt).strip().replace(",", "")
        return int(val, 0)  # handles 0x, 0o, 0b prefixes
    except ValueError:
        print("  Invalid integer.")
        return None


def main() -> None:
    print("Number Base Converter")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            n = get_int("  Decimal number: ")
            if n is None:
                continue
            results = convert_all(n)
            print()
            for label, val in results.items():
                print(f"  {label:<20}: {val}")

        elif choice == "2":
            n = get_int("  Decimal number: ")
            if n is None:
                continue
            base_s = input("  Target base (2-36): ").strip()
            try:
                base = int(base_s)
                result = to_base(n, base)
                print(f"\n  {n} in base {base} = {result}")
            except (ValueError, Exception) as e:
                print(f"  Error: {e}")

        elif choice == "3":
            s = input("  Number string: ").strip().upper()
            base_s = input("  Source base (2-36): ").strip()
            try:
                base = int(base_s)
                n = from_base(s, base)
                print(f"\n  Decimal = {n}")
                print(f"  Hex     = {hex(n).upper()}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "4":
            n = get_int("  Decimal number: ")
            if n is None:
                continue
            base_s = input("  Target base: ").strip()
            try:
                base = int(base_s)
                steps = conversion_steps(n, base)
                print(f"\n  Dividing {abs(n)} by {base}:")
                for step in steps:
                    print(step)
                result = to_base(n, base)
                print(f"\n  Reading remainders bottom-up: {result}")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "5":
            n = get_int("  Integer: ")
            if n is None:
                continue
            bits_s = input("  Bits (8/16/32, default 8): ").strip() or "8"
            try:
                bits = int(bits_s)
                tc = twos_complement(n, bits)
                print(f"\n  {n} in {bits}-bit two's complement: {tc}")
                print(f"  Grouped: {' '.join(tc[i:i+4] for i in range(0,len(tc),4))}")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "6":
            start_s = input("  Start ASCII code (default 32): ").strip() or "32"
            end_s   = input("  End   ASCII code (default 127): ").strip() or "127"
            try:
                lines = ascii_table(int(start_s), int(end_s))
                for line in lines:
                    print(line)
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "7":
            f_s = input("  Float value: ").strip()
            try:
                f = float(f_s)
                info = ieee754(f)
                print()
                for k, v in info.items():
                    print(f"  {k:<14}: {v}")
            except ValueError:
                print("  Invalid float.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
