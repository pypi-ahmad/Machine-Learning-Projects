"""Tip Calculator — CLI tool.

Calculates tip amount, total bill, and per-person split.
Supports custom tip percentages and multiple split scenarios.

Usage:
    python main.py
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMMON_TIP_PERCENTAGES = [10, 15, 18, 20, 25]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def calculate_tip(bill: float, tip_pct: float, num_people: int = 1) -> dict:
    if bill < 0:
        raise ValueError("Bill amount cannot be negative.")
    if tip_pct < 0:
        raise ValueError("Tip percentage cannot be negative.")
    if num_people < 1:
        raise ValueError("Number of people must be at least 1.")

    tip_amount = bill * (tip_pct / 100)
    total = bill + tip_amount
    per_person = total / num_people
    tip_per_person = tip_amount / num_people

    return {
        "bill":           bill,
        "tip_pct":        tip_pct,
        "tip_amount":     tip_amount,
        "total":          total,
        "num_people":     num_people,
        "per_person":     per_person,
        "tip_per_person": tip_per_person,
    }


def round_up_per_person(per_person: float) -> float:
    """Round up to the nearest dollar (avoids awkward change)."""
    import math
    return math.ceil(per_person)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_result(r: dict) -> None:
    print(f"\n  {'─' * 34}")
    print(f"  Bill amount      : ${r['bill']:.2f}")
    print(f"  Tip ({r['tip_pct']:.0f}%)         : ${r['tip_amount']:.2f}")
    print(f"  Total            : ${r['total']:.2f}")
    if r['num_people'] > 1:
        print(f"  People           : {r['num_people']}")
        print(f"  Per person       : ${r['per_person']:.2f}  (tip ${r['tip_per_person']:.2f})")
        rounded = round_up_per_person(r['per_person'])
        print(f"  Rounded up/pp    : ${rounded:.0f}")
    print(f"  {'─' * 34}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def get_positive_float(prompt: str) -> float:
    while True:
        try:
            val = float(input(prompt).strip())
            if val >= 0:
                return val
            print("  Value cannot be negative.")
        except ValueError:
            print("  Please enter a valid number.")


def get_positive_int(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if val >= 1:
                return val
            print("  Must be at least 1.")
        except ValueError:
            print("  Please enter a whole number.")


def main() -> None:
    print("Tip Calculator")
    print("=" * 40)

    while True:
        print("\n1. Calculate tip")
        print("2. Show tip table (multiple percentages)")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            bill = get_positive_float("  Bill amount ($): ")
            tip_str = input(f"  Tip percentage (default 18): ").strip()
            tip_pct = float(tip_str) if tip_str else 18.0
            people = get_positive_int("  Number of people (default 1): ") if \
                input("  Split the bill? (y/n, default n): ").strip().lower() == "y" else 1

            try:
                result = calculate_tip(bill, tip_pct, people)
                display_result(result)
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "2":
            bill = get_positive_float("  Bill amount ($): ")
            people_str = input("  Number of people (default 1): ").strip()
            people = int(people_str) if people_str.isdigit() else 1
            print(f"\n  {'Tip %':<8} {'Tip $':<10} {'Total':<10} {'Per Person'}")
            print(f"  {'─' * 44}")
            for pct in COMMON_TIP_PERCENTAGES:
                r = calculate_tip(bill, pct, people)
                pp = f"${r['per_person']:.2f}" if people > 1 else "—"
                print(f"  {pct:<8} ${r['tip_amount']:<9.2f} ${r['total']:<9.2f} {pp}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
