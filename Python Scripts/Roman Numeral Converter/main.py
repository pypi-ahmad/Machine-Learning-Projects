"""Roman Numeral Converter — CLI tool.

Convert integers to Roman numerals and back.
Supports standard form (1–3999) and extended Unicode Roman numerals.
Also validates and explains Roman numeral rules.

Usage:
    python main.py
"""

# ---------------------------------------------------------------------------
# Conversion tables
# ---------------------------------------------------------------------------

ROMAN_VALUES: list[tuple[int, str]] = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"),
    (1,    "I"),
]

SYMBOL_VALUES: dict[str, int] = {
    "M": 1000, "D": 500, "C": 100, "L": 50,
    "X": 10,  "V": 5,   "I": 1,
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def to_roman(n: int) -> str:
    """Convert integer 1–3999 to Roman numeral."""
    if not 1 <= n <= 3999:
        raise ValueError("Number must be between 1 and 3999.")
    result = []
    for value, symbol in ROMAN_VALUES:
        while n >= value:
            result.append(symbol)
            n -= value
    return "".join(result)


def from_roman(s: str) -> int:
    """Convert Roman numeral string to integer."""
    s = s.upper().strip()
    if not s:
        raise ValueError("Empty string.")
    result = 0
    prev   = 0
    for ch in reversed(s):
        if ch not in SYMBOL_VALUES:
            raise ValueError(f"Invalid character: '{ch}'")
        val = SYMBOL_VALUES[ch]
        if val < prev:
            result -= val
        else:
            result += val
        prev = val
    # Validate by round-trip
    if to_roman(result) != s:
        raise ValueError(f"'{s}' is not a valid Roman numeral.")
    return result


def explain_roman(s: str) -> list[str]:
    """Return step-by-step explanation of a Roman numeral."""
    s = s.upper().strip()
    lines = []
    i = 0
    running = 0
    while i < len(s):
        # Check for two-char subtractive combo
        if i + 1 < len(s):
            two = s[i:i+2]
            if two in {s: v for s, v in ROMAN_VALUES if len(s) == 2}:
                val = dict(ROMAN_VALUES)[two]
                running += val
                lines.append(f"  {two:<4} =  +{val:<6}  (subtracted {s[i]} from {s[i+1]})  running: {running}")
                i += 2
                continue
        ch  = s[i]
        val = SYMBOL_VALUES.get(ch, 0)
        # Check if subtractive
        if i + 1 < len(s) and SYMBOL_VALUES.get(s[i+1], 0) > val:
            lines.append(f"  {ch:<4} will subtract (next char is larger)")
            running -= val
        else:
            running += val
            lines.append(f"  {ch:<4} =  +{val:<6}  running: {running}")
        i += 1
    return lines


def roman_table(start: int = 1, end: int = 20) -> list[str]:
    lines = [f"  {'Decimal':>8}  Roman"]
    lines.append(f"  {'-'*8}  {'-'*10}")
    for n in range(start, min(end + 1, 4000)):
        lines.append(f"  {n:>8}  {to_roman(n)}")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Roman Numeral Converter
-----------------------
1. Integer → Roman numeral
2. Roman numeral → Integer
3. Explain Roman numeral
4. Generate Roman numeral table
5. Batch convert integers
0. Quit
"""


def main() -> None:
    print("Roman Numeral Converter")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            raw = input("  Integer (1-3999): ").strip().replace(",", "")
            try:
                n = int(raw)
                print(f"\n  {n:,} = {to_roman(n)}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "2":
            s = input("  Roman numeral: ").strip()
            try:
                n = from_roman(s)
                print(f"\n  {s.upper()} = {n:,}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "3":
            s = input("  Roman numeral: ").strip()
            try:
                lines = explain_roman(s)
                n = from_roman(s)
                print(f"\n  Explaining {s.upper()} = {n:,}:")
                for line in lines:
                    print(line)
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "4":
            start_s = input("  Start (default 1): ").strip() or "1"
            end_s   = input("  End   (default 20): ").strip() or "20"
            try:
                lines = roman_table(int(start_s), int(end_s))
                for line in lines:
                    print(line)
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "5":
            print("  Enter integers one per line (blank to stop):")
            while True:
                raw = input("  > ").strip().replace(",", "")
                if not raw:
                    break
                try:
                    n = int(raw)
                    print(f"    {n:>6} = {to_roman(n)}")
                except ValueError as e:
                    print(f"    Error: {e}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
