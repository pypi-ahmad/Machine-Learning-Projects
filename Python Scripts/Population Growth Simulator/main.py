"""Population Growth Simulator — CLI tool.

Simulate exponential, logistic, and Malthusian population growth.
Supports carrying capacity, birth/death rates, and ASCII time series.

Usage:
    python main.py
    python main.py --model exponential --p0 1000 --r 0.03 --years 50
    python main.py --model logistic    --p0 100  --r 0.1  --K 10000 --years 100
"""

import argparse
import math
import sys


# ── Models ────────────────────────────────────────────────────────────────────

def exponential_growth(p0: float, r: float, years: int) -> list[float]:
    """P(t) = P0 * e^(r*t)"""
    return [p0 * math.exp(r * t) for t in range(years + 1)]


def logistic_growth(p0: float, r: float, K: float, years: int) -> list[float]:
    """Verhulst logistic model: dP/dt = r*P*(1 - P/K), solved via Euler."""
    pop  = [p0]
    dt   = 0.1
    steps_per_year = int(1 / dt)
    P    = p0
    for _ in range(years * steps_per_year):
        P += dt * r * P * (1 - P / K)
        P  = max(P, 0)
    # Re-run to collect yearly snapshots
    pop = [p0]
    P   = p0
    for yr in range(years):
        for _ in range(steps_per_year):
            P += dt * r * P * (1 - P / K)
            P  = max(P, 0)
        pop.append(P)
    return pop


def discrete_growth(p0: float, birth_rate: float, death_rate: float,
                    years: int, K: float = None) -> list[float]:
    """Discrete-time model with births and deaths; optional carrying capacity."""
    pop = [p0]
    P   = p0
    for _ in range(years):
        births = birth_rate * P
        deaths = death_rate * P
        if K:
            deaths += (P / K) * P * 0.01    # density-dependent mortality
        P = max(P + births - deaths, 0)
        pop.append(P)
    return pop


# ── Display ───────────────────────────────────────────────────────────────────

def ascii_plot(populations: list[float], years: int, title: str,
               width: int = 60, height: int = 20) -> None:
    p_min  = min(populations)
    p_max  = max(populations)
    p_range = p_max - p_min or 1

    grid = [[" "] * width for _ in range(height)]
    for t, p in enumerate(populations):
        col = int(t / years * (width - 1))
        row = int((p_max - p) / p_range * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "●"

    print(f"\n  {title}")
    print(f"  {'─'*width}")
    print(f"  {p_max:>12,.0f}")
    for row in grid:
        print("  |" + "".join(row))
    print(f"  └{'─'*width}")
    print(f"  Year 0{' '*(width-12)}Year {years}")
    print(f"  {p_min:>12,.0f}")


def print_table(populations: list[float], label: str = "Population",
                every: int = 10) -> None:
    n = len(populations) - 1
    print(f"\n  {'Year':>6}  {label:>16}  {'Growth':>10}  {'Rate':>8}")
    print("  " + "─" * 46)
    for t in range(0, n + 1, max(1, every)):
        p = populations[t]
        g = p - populations[t - every] if t >= every else 0
        r = g / populations[t - every] if t >= every and populations[t - every] else 0
        g_str = f"+{g:,.0f}" if g >= 0 else f"{g:,.0f}"
        r_str = f"{r:.2%}" if t else "—"
        print(f"  {t:>6}  {p:>16,.1f}  {g_str:>10}  {r_str:>8}")


def summarize(populations: list[float], model_name: str,
              p0: float, K: float = None) -> None:
    final   = populations[-1]
    max_p   = max(populations)
    total_g = final / p0 if p0 else 0

    print(f"\n  Model:           {model_name}")
    print(f"  Initial pop:     {p0:,.0f}")
    print(f"  Final pop:       {final:,.0f}")
    print(f"  Peak pop:        {max_p:,.0f}")
    print(f"  Total growth:    {total_g:.2f}×")
    if K:
        print(f"  Carrying cap:    {K:,.0f}")
        print(f"  Saturation:      {final/K:.1%}")


# ── Interactive ───────────────────────────────────────────────────────────────

def interactive():
    print("=== Population Growth Simulator ===")
    print("Models: exponential | logistic | discrete | quit\n")
    while True:
        cmd = input("Model: ").strip().lower()
        if cmd in ("quit", "q", "exit"): break

        try:
            p0    = float(input("  Initial population: ").strip())
            years = int(input("  Number of years: ").strip())

            if cmd == "exponential":
                r  = float(input("  Growth rate r (e.g. 0.03 for 3%/yr): ").strip())
                pop = exponential_growth(p0, r, years)
                title = f"Exponential Growth  r={r}"
            elif cmd == "logistic":
                r  = float(input("  Growth rate r: ").strip())
                K  = float(input("  Carrying capacity K: ").strip())
                pop = logistic_growth(p0, r, K, years)
                title = f"Logistic Growth  r={r}  K={K:,.0f}"
                summarize(pop, "Logistic", p0, K)
            elif cmd == "discrete":
                b = float(input("  Birth rate (per individual per year): ").strip())
                d = float(input("  Death rate (per individual per year): ").strip())
                K = input("  Carrying capacity (leave blank for none): ").strip()
                K = float(K) if K else None
                pop = discrete_growth(p0, b, d, years, K)
                title = f"Discrete Growth  b={b}  d={d}"
            else:
                print("  Unknown model."); continue

            ascii_plot(pop, years, title)
            print_table(pop, every=max(1, years // 10))
            if cmd != "logistic":
                summarize(pop, cmd.capitalize(), p0)
        except ValueError as e:
            print(f"  Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Population Growth Simulator")
    parser.add_argument("--model",  choices=["exponential","logistic","discrete"],
                        default=None)
    parser.add_argument("--p0",     type=float, default=1000, help="Initial population")
    parser.add_argument("--r",      type=float, default=0.05, help="Growth rate")
    parser.add_argument("--K",      type=float, default=None, help="Carrying capacity")
    parser.add_argument("--years",  type=int,   default=50,   help="Simulation years")
    parser.add_argument("--birth",  type=float, default=0.08, help="Birth rate (discrete)")
    parser.add_argument("--death",  type=float, default=0.03, help="Death rate (discrete)")
    args = parser.parse_args()

    if args.model == "exponential":
        pop = exponential_growth(args.p0, args.r, args.years)
        ascii_plot(pop, args.years, f"Exponential Growth r={args.r}")
        print_table(pop, every=max(1, args.years // 10))
        summarize(pop, "Exponential", args.p0)
    elif args.model == "logistic":
        if not args.K:
            print("--K required for logistic model."); sys.exit(1)
        pop = logistic_growth(args.p0, args.r, args.K, args.years)
        ascii_plot(pop, args.years, f"Logistic Growth r={args.r} K={args.K:,.0f}")
        print_table(pop, every=max(1, args.years // 10))
        summarize(pop, "Logistic", args.p0, args.K)
    elif args.model == "discrete":
        pop = discrete_growth(args.p0, args.birth, args.death, args.years, args.K)
        ascii_plot(pop, args.years, f"Discrete Growth b={args.birth} d={args.death}")
        print_table(pop, every=max(1, args.years // 10))
        summarize(pop, "Discrete", args.p0, args.K)
    else:
        interactive()


if __name__ == "__main__":
    main()
