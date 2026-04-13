"""Percentage Calculator — CLI tool.

Solves the most common percentage problems:
  1. What is X% of Y?
  2. X is what % of Y?
  3. Percentage change from X to Y
  4. Add X% to Y (markup)
  5. Subtract X% from Y (discount)

Usage:
    python main.py
"""

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def percent_of(pct: float, value: float) -> float:
    """What is pct% of value?"""
    return (pct / 100) * value


def what_percent(part: float, whole: float) -> float:
    """Part is what percent of whole?"""
    if whole == 0:
        raise ValueError("Whole cannot be zero.")
    return (part / whole) * 100


def percent_change(old: float, new: float) -> float:
    """Percentage change from old to new."""
    if old == 0:
        raise ValueError("Original value cannot be zero.")
    return ((new - old) / abs(old)) * 100


def add_percent(value: float, pct: float) -> float:
    """Add pct% to value (markup)."""
    return value + percent_of(pct, value)


def subtract_percent(value: float, pct: float) -> float:
    """Subtract pct% from value (discount)."""
    return value - percent_of(pct, value)


# ---------------------------------------------------------------------------
# Menu and CLI
# ---------------------------------------------------------------------------

MENU = """
Percentage Calculator
---------------------
1. What is X% of Y?
2. X is what % of Y?
3. Percentage change from X to Y
4. Add X% to Y  (markup / increase)
5. Subtract X% from Y  (discount / decrease)
0. Quit
"""


def get_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  Please enter a valid number.")


def main() -> None:
    print("Percentage Calculator")
    while True:
        print(MENU)
        choice = input("Choose (0-5): ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            pct = get_float("  Enter percentage (X): ")
            val = get_float("  Enter value (Y): ")
            result = percent_of(pct, val)
            print(f"\n  {pct}% of {val} = {result:.4g}")

        elif choice == "2":
            part = get_float("  Enter part (X): ")
            whole = get_float("  Enter whole (Y): ")
            try:
                result = what_percent(part, whole)
                print(f"\n  {part} is {result:.4g}% of {whole}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "3":
            old = get_float("  Enter original value (X): ")
            new = get_float("  Enter new value (Y): ")
            try:
                change = percent_change(old, new)
                direction = "increase" if change >= 0 else "decrease"
                print(f"\n  Change from {old} to {new}: {abs(change):.4g}% {direction}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "4":
            val = get_float("  Enter base value (Y): ")
            pct = get_float("  Enter percentage to add (X): ")
            result = add_percent(val, pct)
            added = percent_of(pct, val)
            print(f"\n  {val} + {pct}% = {result:.4g}  (added {added:.4g})")

        elif choice == "5":
            val = get_float("  Enter base value (Y): ")
            pct = get_float("  Enter percentage to subtract (X): ")
            result = subtract_percent(val, pct)
            removed = percent_of(pct, val)
            print(f"\n  {val} - {pct}% = {result:.4g}  (removed {removed:.4g})")

        else:
            print("  Invalid choice. Enter 0-5.")


if __name__ == "__main__":
    main()
