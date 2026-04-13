"""Loan EMI Calculator — CLI tool.

Calculates monthly EMI (Equated Monthly Installment),
total payment, total interest, and an amortization schedule.

Formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
  P = principal, r = monthly rate, n = months

Usage:
    python main.py
"""

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> dict:
    if principal <= 0:
        raise ValueError("Principal must be positive.")
    if annual_rate_pct < 0:
        raise ValueError("Interest rate cannot be negative.")
    if tenure_months <= 0:
        raise ValueError("Tenure must be at least 1 month.")

    monthly_rate = annual_rate_pct / 12 / 100

    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        factor = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * factor / (factor - 1)

    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    return {
        "principal":      principal,
        "annual_rate":    annual_rate_pct,
        "monthly_rate":   monthly_rate,
        "tenure_months":  tenure_months,
        "emi":            emi,
        "total_payment":  total_payment,
        "total_interest": total_interest,
    }


def amortization_schedule(result: dict) -> list[dict]:
    balance = result["principal"]
    rate = result["monthly_rate"]
    emi = result["emi"]
    rows = []
    for month in range(1, result["tenure_months"] + 1):
        interest = balance * rate
        principal_paid = emi - interest
        balance -= principal_paid
        if balance < 0:
            balance = 0.0
        rows.append({
            "month":          month,
            "emi":            emi,
            "principal_paid": principal_paid,
            "interest_paid":  interest,
            "balance":        balance,
        })
    return rows


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_summary(r: dict) -> None:
    years = r["tenure_months"] // 12
    months = r["tenure_months"] % 12
    tenure_str = f"{years}y {months}m" if years else f"{months}m"
    print(f"\n  {'─' * 40}")
    print(f"  Principal        : ${r['principal']:>12,.2f}")
    print(f"  Annual rate      : {r['annual_rate']:.2f}%")
    print(f"  Tenure           : {r['tenure_months']} months ({tenure_str})")
    print(f"  Monthly EMI      : ${r['emi']:>12,.2f}")
    print(f"  Total payment    : ${r['total_payment']:>12,.2f}")
    print(f"  Total interest   : ${r['total_interest']:>12,.2f}")
    interest_pct = (r['total_interest'] / r['principal']) * 100
    print(f"  Interest / loan  : {interest_pct:.1f}%")
    print(f"  {'─' * 40}")


def display_schedule(schedule: list[dict], max_rows: int = 24) -> None:
    print(f"\n  {'Month':<7} {'EMI':>10} {'Principal':>12} {'Interest':>12} {'Balance':>14}")
    print(f"  {'─' * 58}")
    for row in schedule[:max_rows]:
        print(f"  {row['month']:<7} ${row['emi']:>9,.2f} ${row['principal_paid']:>11,.2f}"
              f" ${row['interest_paid']:>11,.2f} ${row['balance']:>13,.2f}")
    if len(schedule) > max_rows:
        print(f"  ... ({len(schedule) - max_rows} more rows hidden)")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def get_positive_float(prompt: str) -> float:
    while True:
        try:
            val = float(input(prompt).strip())
            if val > 0:
                return val
            print("  Must be > 0.")
        except ValueError:
            print("  Please enter a valid number.")


def get_positive_int(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if val > 0:
                return val
            print("  Must be > 0.")
        except ValueError:
            print("  Please enter a whole number.")


def main() -> None:
    print("Loan EMI Calculator")
    print("=" * 40)

    while True:
        print("\n1. Calculate EMI")
        print("2. Compare two loans side by side")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            principal = get_positive_float("  Loan amount ($): ")
            rate = get_positive_float("  Annual interest rate (%): ")
            tenure_input = input("  Tenure in years (or add 'm' for months, e.g. '5' or '60m'): ").strip()
            if tenure_input.lower().endswith("m"):
                tenure = int(tenure_input[:-1])
            else:
                tenure = int(float(tenure_input) * 12)

            try:
                result = calculate_emi(principal, rate, tenure)
                display_summary(result)
                if input("\n  Show amortization schedule? (y/n): ").strip().lower() == "y":
                    schedule = amortization_schedule(result)
                    display_schedule(schedule)
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "2":
            print("\n  --- Loan A ---")
            pA = get_positive_float("  Principal ($): ")
            rA = get_positive_float("  Annual rate (%): ")
            tA = int(get_positive_float("  Tenure (years): ") * 12)

            print("\n  --- Loan B ---")
            pB = get_positive_float("  Principal ($): ")
            rB = get_positive_float("  Annual rate (%): ")
            tB = int(get_positive_float("  Tenure (years): ") * 12)

            try:
                A = calculate_emi(pA, rA, tA)
                B = calculate_emi(pB, rB, tB)
                print(f"\n  {'':20} {'Loan A':>14} {'Loan B':>14}")
                print(f"  {'─' * 50}")
                print(f"  {'Principal':<20} ${A['principal']:>13,.2f} ${B['principal']:>13,.2f}")
                print(f"  {'Annual rate':<20} {A['annual_rate']:>13.2f}% {B['annual_rate']:>13.2f}%")
                print(f"  {'Tenure (months)':<20} {A['tenure_months']:>14} {B['tenure_months']:>14}")
                print(f"  {'Monthly EMI':<20} ${A['emi']:>13,.2f} ${B['emi']:>13,.2f}")
                print(f"  {'Total payment':<20} ${A['total_payment']:>13,.2f} ${B['total_payment']:>13,.2f}")
                print(f"  {'Total interest':<20} ${A['total_interest']:>13,.2f} ${B['total_interest']:>13,.2f}")
            except ValueError as e:
                print(f"  Error: {e}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
