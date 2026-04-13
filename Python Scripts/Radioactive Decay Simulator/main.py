"""Radioactive Decay Simulator — CLI tool.

Simulate radioactive decay chains, compute half-life, activity,
and visualize decay curves with ASCII plots.

Usage:
    python main.py
    python main.py --element C-14 --n0 1000 --years 10000
    python main.py --halflife 5730 --n0 1e12 --years 20000
"""

import argparse
import math
import sys


# ── Known isotopes ────────────────────────────────────────────────────────────
# (half-life in seconds)
ISOTOPES = {
    "H-3":    {"name": "Tritium",       "hl_sec": 3.888e8,  "hl_str": "12.32 yr"},
    "C-14":   {"name": "Carbon-14",     "hl_sec": 1.807e11, "hl_str": "5,730 yr"},
    "Co-60":  {"name": "Cobalt-60",     "hl_sec": 1.664e8,  "hl_str": "5.27 yr"},
    "Sr-90":  {"name": "Strontium-90",  "hl_sec": 9.093e8,  "hl_str": "28.8 yr"},
    "Cs-137": {"name": "Cesium-137",    "hl_sec": 9.487e8,  "hl_str": "30.1 yr"},
    "Ra-226": {"name": "Radium-226",    "hl_sec": 5.049e10, "hl_str": "1,600 yr"},
    "U-235":  {"name": "Uranium-235",   "hl_sec": 2.221e16, "hl_str": "703.8 Myr"},
    "U-238":  {"name": "Uranium-238",   "hl_sec": 1.409e17, "hl_str": "4.47 Gyr"},
    "Pu-239": {"name": "Plutonium-239", "hl_sec": 7.617e11, "hl_str": "24,100 yr"},
    "I-131":  {"name": "Iodine-131",    "hl_sec": 6.948e5,  "hl_str": "8.02 d"},
    "Tc-99m": {"name": "Technetium-99m","hl_sec": 2.163e4,  "hl_str": "6.01 h"},
    "F-18":   {"name": "Fluorine-18",   "hl_sec": 6.584e3,  "hl_str": "109.7 min"},
}


def decay_constant(half_life_sec: float) -> float:
    return math.log(2) / half_life_sec


def simulate(N0: float, lambda_: float, times: list[float]) -> list[float]:
    """N(t) = N0 * exp(-lambda * t)"""
    return [N0 * math.exp(-lambda_ * t) for t in times]


def activity(N: float, lambda_: float) -> float:
    """Activity A = lambda * N  (decays per second)"""
    return lambda_ * N


def half_lives_to_fraction(n: float) -> float:
    return 0.5 ** n


def fraction_remaining(elapsed: float, half_life: float) -> float:
    return 0.5 ** (elapsed / half_life)


def time_for_fraction(frac: float, half_life: float) -> float:
    """Time to reach given fraction of original."""
    if frac <= 0 or frac > 1:
        raise ValueError("Fraction must be in (0, 1].")
    return -math.log(frac) / math.log(2) * half_life


def ascii_decay_plot(N0: float, lambda_: float, half_life_years: float,
                     years: int, width: int = 60, height: int = 18) -> None:
    n_points = width
    times_yr = [i / (n_points - 1) * years for i in range(n_points)]
    times_s  = [t * 3.156e7 for t in times_yr]
    pops     = simulate(N0, lambda_, times_s)

    grid = [[" "] * width for _ in range(height)]
    p_max = pops[0]
    for col, p in enumerate(pops):
        row = int((1 - p / p_max) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "●"

    # Mark half-life boundaries
    hl_cols = []
    for k in range(1, 6):
        t_hl = k * half_life_years
        if t_hl <= years:
            col = int(t_hl / years * (width - 1))
            hl_cols.append(col)
            for row in range(height):
                if grid[row][col] == " ":
                    grid[row][col] = "┆"

    print(f"\n  {'─'*width}")
    print(f"  {N0:>12,.2g}")
    for row in grid:
        print("  |" + "".join(row))
    print(f"  └{'─'*width}")
    print(f"  Year 0{' '*(width-12)}Year {years:,}")
    if hl_cols:
        markers = " ".join(f"t½×{k+1}" for k in range(len(hl_cols)))
        print(f"  ┆ = half-life boundaries: {markers}")


def display_info(symbol: str, N0: float, total_years: int) -> None:
    iso     = ISOTOPES[symbol]
    hl_sec  = iso["hl_sec"]
    hl_yr   = hl_sec / 3.156e7
    lam     = decay_constant(hl_sec)

    print(f"\n  Isotope:       {symbol} — {iso['name']}")
    print(f"  Half-life:     {iso['hl_str']}")
    print(f"  λ (decay const): {lam:.4e} s⁻¹")
    print(f"  Initial N₀:    {N0:,.2g}")
    A0 = activity(N0, lam)
    print(f"  Initial activity: {A0:,.4e} Bq (decays/s)")

    print(f"\n  {'Years':>12}  {'N remaining':>16}  {'% remaining':>12}  {'Activity (Bq)':>16}")
    print("  " + "─" * 62)
    for yr in [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000,
               total_years]:
        if yr > total_years and yr != total_years: continue
        t  = yr * 3.156e7
        N  = N0 * math.exp(-lam * t)
        A  = activity(N, lam)
        pc = N / N0 * 100
        print(f"  {yr:>12,}  {N:>16,.4e}  {pc:>11.4f}%  {A:>16,.4e}")

    ascii_decay_plot(N0, lam, hl_yr, total_years)


def custom_decay(N0: float, hl_years: float, total_years: int) -> None:
    hl_sec = hl_years * 3.156e7
    lam    = decay_constant(hl_sec)
    print(f"\n  Half-life:  {hl_years:,.2f} years")
    print(f"  λ:          {lam:.4e} s⁻¹")
    print(f"  Initial N₀: {N0:,.2g}")
    print(f"\n  {'Years':>12}  {'N remaining':>16}  {'% remaining':>12}")
    print("  " + "─" * 44)
    for yr in [0] + list(range(total_years // 10, total_years + 1, total_years // 10)):
        t  = yr * 3.156e7
        N  = N0 * math.exp(-lam * t)
        pc = N / N0 * 100
        print(f"  {yr:>12,}  {N:>16,.4e}  {pc:>11.4f}%")
    ascii_decay_plot(N0, lam, hl_years, total_years)


def interactive():
    print("=== Radioactive Decay Simulator ===")
    print(f"Known isotopes: {', '.join(ISOTOPES.keys())}")
    print("Commands: isotope | custom | list | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "list":
            print(f"\n  {'Symbol':>8}  {'Name':>20}  {'Half-life':>14}")
            for sym, d in ISOTOPES.items():
                print(f"  {sym:>8}  {d['name']:>20}  {d['hl_str']:>14}")
        elif cmd == "isotope":
            sym   = input("  Symbol (e.g. C-14): ").strip()
            if sym not in ISOTOPES:
                print(f"  Unknown. Available: {', '.join(ISOTOPES.keys())}"); continue
            N0    = float(input("  Initial atoms N₀: ").strip())
            years = int(input("  Simulation years: ").strip())
            display_info(sym, N0, years)
        elif cmd == "custom":
            N0    = float(input("  Initial atoms N₀: ").strip())
            hl    = float(input("  Half-life (years): ").strip())
            years = int(input("  Simulation years: ").strip())
            custom_decay(N0, hl, years)
        else:
            print("  Commands: isotope | custom | list | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Radioactive Decay Simulator")
    parser.add_argument("--element",  metavar="SYM",  help=f"Isotope symbol ({', '.join(ISOTOPES.keys())})")
    parser.add_argument("--halflife", type=float, metavar="YR", help="Custom half-life in years")
    parser.add_argument("--n0",       type=float, default=1e12,  help="Initial number of atoms")
    parser.add_argument("--years",    type=int,   default=10000, help="Simulation years")
    args = parser.parse_args()

    if args.element:
        sym = args.element.upper()
        if sym not in ISOTOPES:
            print(f"Unknown isotope '{sym}'. Available: {', '.join(ISOTOPES.keys())}")
            sys.exit(1)
        display_info(sym, args.n0, args.years)
    elif args.halflife:
        custom_decay(args.n0, args.halflife, args.years)
    else:
        interactive()


if __name__ == "__main__":
    main()
