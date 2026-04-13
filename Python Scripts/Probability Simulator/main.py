"""Probability Simulator — CLI tool.

Simulate common probability experiments:
coin flips, dice rolls, card draws, birthday problem, Monty Hall.

Usage:
    python main.py
    python main.py --coin 1000
    python main.py --dice 2 10000
    python main.py --birthday 23 10000
    python main.py --monty 10000
"""

import argparse
import random


# ── Experiments ───────────────────────────────────────────────────────────────

def sim_coins(n_flips: int, n_trials: int) -> None:
    """Simulate n_flips fair coin flips n_trials times and report distributions."""
    from collections import Counter
    head_counts = Counter()
    for _ in range(n_trials):
        heads = sum(random.randint(0, 1) for _ in range(n_flips))
        head_counts[heads] += 1

    print(f"\nCoin flip simulation: {n_flips} flip(s) × {n_trials:,} trials")
    print(f"{'Heads':>6}  {'Count':>8}  {'Probability':>12}")
    for k in range(n_flips + 1):
        p = head_counts[k] / n_trials
        bar = "█" * int(p * 40)
        print(f"{k:>6}  {head_counts[k]:>8,}  {p:>12.4f}  {bar}")


def sim_dice(n_dice: int, n_rolls: int) -> None:
    """Simulate rolling n_dice six-sided dice n_rolls times."""
    from collections import Counter
    sums = Counter()
    for _ in range(n_rolls):
        sums[sum(random.randint(1, 6) for _ in range(n_dice))] += 1

    min_sum = n_dice
    max_sum = n_dice * 6
    print(f"\nDice simulation: {n_dice}d6 × {n_rolls:,} rolls")
    print(f"{'Sum':>5}  {'Count':>8}  {'Probability':>12}")
    for s in range(min_sum, max_sum + 1):
        p = sums[s] / n_rolls
        bar = "█" * int(p * 50)
        print(f"{s:>5}  {sums[s]:>8,}  {p:>12.4f}  {bar}")


def sim_birthday(n_people: int, n_trials: int) -> None:
    """Birthday problem: probability that ≥2 people share a birthday."""
    matches = 0
    for _ in range(n_trials):
        bdays = [random.randint(1, 365) for _ in range(n_people)]
        if len(bdays) != len(set(bdays)):
            matches += 1
    p_sim = matches / n_trials

    # Exact theoretical probability
    p_none = 1.0
    for k in range(n_people):
        p_none *= (365 - k) / 365
    p_theory = 1 - p_none

    print(f"\nBirthday problem: {n_people} people, {n_trials:,} trials")
    print(f"  Simulated probability of shared birthday: {p_sim:.4f} ({p_sim*100:.2f}%)")
    print(f"  Theoretical probability:                  {p_theory:.4f} ({p_theory*100:.2f}%)")


def sim_monty_hall(n_trials: int) -> None:
    """Monty Hall problem: compare switch vs stay strategies."""
    switch_wins = 0
    stay_wins   = 0
    for _ in range(n_trials):
        car   = random.randint(0, 2)
        pick  = random.randint(0, 2)
        # Host opens a goat door (not car, not pick)
        goats = [d for d in range(3) if d != car and d != pick]
        open_ = random.choice(goats)
        # Switch
        switch = next(d for d in range(3) if d != pick and d != open_)
        if switch == car: switch_wins += 1
        if pick   == car: stay_wins   += 1

    print(f"\nMonty Hall simulation: {n_trials:,} trials")
    print(f"  Switch strategy wins: {switch_wins:,}/{n_trials:,} = {switch_wins/n_trials:.4f} "
          f"(theory: 2/3 ≈ 0.6667)")
    print(f"  Stay   strategy wins: {stay_wins:,}/{n_trials:,} = {stay_wins/n_trials:.4f} "
          f"(theory: 1/3 ≈ 0.3333)")


def sim_cards(n_draws: int, n_trials: int) -> None:
    """Draw n_draws cards from a standard deck, report suit/rank frequencies."""
    from collections import Counter
    suits = ["♠", "♥", "♦", "♣"]
    ranks = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    deck  = [f"{r}{s}" for s in suits for r in ranks]

    suit_counts = Counter()
    for _ in range(n_trials):
        drawn = random.sample(deck, n_draws)
        for card in drawn:
            suit_counts[card[-1]] += 1

    total = sum(suit_counts.values())
    print(f"\nCard draw: {n_draws} card(s) × {n_trials:,} trials")
    print(f"{'Suit':>6}  {'Count':>8}  {'Probability':>12}  {'Expected':>10}")
    for s in suits:
        p = suit_counts[s] / total
        print(f"{s:>6}  {suit_counts[s]:>8,}  {p:>12.4f}  {0.25:>10.4f}")


# ── Interactive ───────────────────────────────────────────────────────────────

def interactive():
    print("=== Probability Simulator ===")
    print("Commands: coin | dice | birthday | monty | cards | quit\n")
    while True:
        cmd = input("Experiment: ").strip().lower()
        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "coin":
            n_flips  = int(input("  Flips per trial [default 1]: ").strip() or 1)
            n_trials = int(input("  Number of trials [default 10000]: ").strip() or 10000)
            sim_coins(n_flips, n_trials)
        elif cmd == "dice":
            n_dice   = int(input("  Number of dice [default 2]: ").strip() or 2)
            n_rolls  = int(input("  Number of rolls [default 10000]: ").strip() or 10000)
            sim_dice(n_dice, n_rolls)
        elif cmd == "birthday":
            n_people = int(input("  People per room [default 23]: ").strip() or 23)
            n_trials = int(input("  Trials [default 10000]: ").strip() or 10000)
            sim_birthday(n_people, n_trials)
        elif cmd == "monty":
            n_trials = int(input("  Trials [default 10000]: ").strip() or 10000)
            sim_monty_hall(n_trials)
        elif cmd == "cards":
            n_draws  = int(input("  Cards to draw [default 5]: ").strip() or 5)
            n_trials = int(input("  Trials [default 10000]: ").strip() or 10000)
            sim_cards(n_draws, n_trials)
        else:
            print("  Unknown command.\n")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Probability Simulator")
    parser.add_argument("--coin",     nargs="?", const=10000, type=int, metavar="TRIALS",
                        help="Coin flip simulation (single flip, N trials)")
    parser.add_argument("--dice",     nargs=2,   type=int, metavar=("NDICE","ROLLS"),
                        help="Dice simulation: ndice nrolls")
    parser.add_argument("--birthday", nargs=2,   type=int, metavar=("N","TRIALS"),
                        help="Birthday problem: n_people n_trials")
    parser.add_argument("--monty",    nargs="?", const=10000, type=int, metavar="TRIALS",
                        help="Monty Hall simulation")
    parser.add_argument("--cards",    nargs=2,   type=int, metavar=("DRAWS","TRIALS"),
                        help="Card draw: n_draws n_trials")
    args = parser.parse_args()

    if   args.coin:     sim_coins(1, args.coin)
    elif args.dice:     sim_dice(*args.dice)
    elif args.birthday: sim_birthday(*args.birthday)
    elif args.monty:    sim_monty_hall(args.monty)
    elif args.cards:    sim_cards(*args.cards)
    else:               interactive()


if __name__ == "__main__":
    main()
