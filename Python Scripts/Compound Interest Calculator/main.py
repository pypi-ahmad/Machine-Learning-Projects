"""Compound Interest Calculator — CLI tool.

Calculate future value, present value, required rate, required time,
and monthly payment (annuity).  Shows year-by-year growth tables.

Usage:
    python main.py
"""

import math


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def future_value(principal: float, rate: float, n: int,
                  t: float, contribution: float = 0.0) -> float:
    """FV = P(1 + r/n)^(nt) + PMT × [((1+r/n)^(nt)-1) / (r/n)]"""
    rn = rate / n
    periods = n * t
    fv_lump = principal * (1 + rn) ** periods
    if contribution == 0 or rate == 0:
        fv_annuity = contribution * periods
    else:
        fv_annuity = contribution * (((1 + rn) ** periods - 1) / rn)
    return fv_lump + fv_annuity


def present_value(fv: float, rate: float, n: int, t: float) -> float:
    """PV = FV / (1 + r/n)^(nt)"""
    return fv / (1 + rate / n) ** (n * t)


def required_rate(pv: float, fv: float, n: int, t: float) -> float:
    """Solve FV = PV(1 + r/n)^(nt) for r."""
    ratio = fv / pv
    return n * (ratio ** (1 / (n * t)) - 1)


def required_time(pv: float, fv: float, rate: float, n: int) -> float:
    """Solve for t."""
    return math.log(fv / pv) / (n * math.log(1 + rate / n))


def emi(principal: float, annual_rate: float, months: int) -> float:
    """Equated Monthly Instalment."""
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def growth_table(principal: float, rate: float, n: int,
                  t: float, contribution: float = 0.0) -> list[dict]:
    rows = []
    for year in range(1, int(t) + 2):
        y = min(year, t)
        fv = future_value(principal, rate, n, y, contribution)
        rows.append({
            "Year": year,
            "Balance": fv,
            "Interest earned": fv - principal - contribution * n * y,
        })
    return rows


def _pct(r: float) -> str:
    return f"{r * 100:.4g}%"


def _money(v: float) -> str:
    return f"${v:,.2f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Compound Interest Calculator
-----------------------------
1. Future Value (lump sum + contributions)
2. Present Value
3. Required interest rate
4. Required time to reach goal
5. Loan EMI (monthly payment)
6. Investment growth table
0. Quit
"""


def get_float(prompt: str) -> float | None:
    try:
        return float(input(prompt).strip().replace(",", "").replace("%", ""))
    except ValueError:
        print("  Invalid number.")
        return None


def get_compounding() -> int:
    print("  Compounding: 1=Annually 2=Semi-annually 4=Quarterly 12=Monthly 365=Daily")
    n_s = input("  n (default 12): ").strip() or "12"
    try:
        return int(n_s)
    except ValueError:
        return 12


def main() -> None:
    print("Compound Interest Calculator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            p    = get_float("  Principal ($): ")
            r    = get_float("  Annual rate (e.g. 8 for 8%): ")
            if r is not None:
                r /= 100
            n    = get_compounding()
            t    = get_float("  Time (years): ")
            cont = get_float("  Regular contribution per period (0=none): ") or 0.0
            if None in (p, r, t):
                continue
            fv = future_value(p, r, n, t, cont)
            total_contrib = cont * n * t
            interest = fv - p - total_contrib
            print(f"\n  Principal    : {_money(p)}")
            print(f"  Contributions: {_money(total_contrib)}")
            print(f"  Interest     : {_money(interest)}")
            print(f"  Future Value : {_money(fv)}")
            print(f"  Total return : {(fv/p - 1)*100:.2f}%")

        elif choice == "2":
            fv   = get_float("  Target future value ($): ")
            r    = get_float("  Annual rate (%): ")
            if r is not None:
                r /= 100
            n    = get_compounding()
            t    = get_float("  Time (years): ")
            if None in (fv, r, t):
                continue
            pv = present_value(fv, r, n, t)
            print(f"\n  Present Value needed: {_money(pv)}")

        elif choice == "3":
            pv = get_float("  Present value ($): ")
            fv = get_float("  Target future value ($): ")
            n  = get_compounding()
            t  = get_float("  Time (years): ")
            if None in (pv, fv, t):
                continue
            r = required_rate(pv, fv, n, t)
            print(f"\n  Required annual rate: {_pct(r)}")

        elif choice == "4":
            pv = get_float("  Present value ($): ")
            fv = get_float("  Target future value ($): ")
            r  = get_float("  Annual rate (%): ")
            if r is not None:
                r /= 100
            n  = get_compounding()
            if None in (pv, fv, r):
                continue
            t = required_time(pv, fv, r, n)
            years  = int(t)
            months = int((t - years) * 12)
            print(f"\n  Required time: {years} years {months} months ({t:.3f} years)")

        elif choice == "5":
            p  = get_float("  Loan amount ($): ")
            r  = get_float("  Annual interest rate (%): ")
            if r is not None:
                r /= 100
            m_s = input("  Loan term (months): ").strip()
            try:
                months = int(m_s)
            except ValueError:
                months = 12
            if None in (p, r):
                continue
            e = emi(p, r, months)
            total_paid = e * months
            interest = total_paid - p
            print(f"\n  Monthly EMI    : {_money(e)}")
            print(f"  Total paid     : {_money(total_paid)}")
            print(f"  Total interest : {_money(interest)}")

        elif choice == "6":
            p    = get_float("  Principal ($): ")
            r    = get_float("  Annual rate (%): ")
            if r is not None:
                r /= 100
            n    = get_compounding()
            t    = get_float("  Years: ")
            cont = get_float("  Contribution per period (0=none): ") or 0.0
            if None in (p, r, t):
                continue
            rows = growth_table(p, r, n, t, cont)
            print(f"\n  {'Year':>6}  {'Balance':>15}  {'Interest Earned':>16}")
            print(f"  {'-'*6}  {'-'*15}  {'-'*16}")
            for row in rows:
                print(f"  {row['Year']:>6}  {_money(row['Balance']):>15}"
                      f"  {_money(row['Interest earned']):>16}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
