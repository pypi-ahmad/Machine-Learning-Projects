"""Dice Roll Simulator — CLI tool.

Roll any combination of dice (d4, d6, d8, d10, d12, d20, d100).
Supports modifiers, advantage/disadvantage, and roll history.

Usage:
    python main.py
    python main.py "2d6+3"
    python main.py "1d20 adv"
    python main.py "4d6 drop1"
"""

import argparse
import random
import re
import sys
from collections import Counter


DICE_FACES = {4: "▲", 6: "⬡", 8: "◆", 10: "●", 12: "✦", 20: "★", 100: "⊙"}
history: list[str] = []


def roll_die(sides: int) -> int:
    return random.randint(1, sides)


def parse_notation(notation: str):
    """Parse NdS[+/-M] notation. Returns (n, sides, modifier)."""
    notation = notation.strip().lower()
    m = re.fullmatch(r"(\d*)d(\d+)\s*([+-]\s*\d+)?", notation)
    if not m:
        raise ValueError(f"Invalid notation: '{notation}'. Use NdS or NdS+M.")
    n     = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    mod   = int(m.group(3).replace(" ", "")) if m.group(3) else 0
    if n < 1 or n > 100:
        raise ValueError("Number of dice must be 1–100.")
    if sides < 2:
        raise ValueError("Die must have at least 2 sides.")
    return n, sides, mod


def do_roll(notation: str, advantage: bool = False, disadvantage: bool = False,
            drop_lowest: int = 0) -> dict:
    """Perform a roll and return detailed result."""
    n, sides, mod = parse_notation(notation)

    if advantage or disadvantage:
        # Roll twice, keep best/worst
        rolls_a = [roll_die(sides) for _ in range(n)]
        rolls_b = [roll_die(sides) for _ in range(n)]
        set_a, set_b = sum(rolls_a), sum(rolls_b)
        if advantage:
            rolls = rolls_a if set_a >= set_b else rolls_b
            label = "ADV"
        else:
            rolls = rolls_a if set_a <= set_b else rolls_b
            label = "DIS"
    else:
        rolls = [roll_die(sides) for _ in range(n)]
        label = ""

    kept   = rolls[:]
    dropped = []
    if drop_lowest and len(kept) > drop_lowest:
        for _ in range(drop_lowest):
            min_val = min(kept)
            dropped.append(min_val)
            kept.remove(min_val)

    total = sum(kept) + mod
    return {
        "notation": notation,
        "n": n, "sides": sides, "mod": mod,
        "rolls": rolls,
        "kept":  kept,
        "dropped": dropped,
        "total": total,
        "label": label,
    }


def display_result(res: dict) -> None:
    face = DICE_FACES.get(res["sides"], "●")
    rolls_str = " ".join(f"[{r}]" for r in res["rolls"])
    if res["dropped"]:
        drops = " ".join(f"~~{r}~~" for r in res["dropped"])
        print(f"  {face} d{res['sides']}  Rolled: {rolls_str}  Dropped: {drops}")
    else:
        print(f"  {face} d{res['sides']}  Rolled: {rolls_str}")
    mod_str = f" {res['mod']:+d}" if res["mod"] else ""
    label   = f"  [{res['label']}]" if res["label"] else ""
    print(f"  Kept: {sum(res['kept'])}{mod_str} = {res['total']}{label}")

    # Min/max highlight
    if res["sides"] in (20, 100) and res["n"] == 1:
        if res["rolls"][0] == res["sides"]:
            print("  🎉 CRITICAL HIT!")
        elif res["rolls"][0] == 1:
            print("  💀 CRITICAL FAIL!")


def stats_display(sides: int, n_rolls: int = 5000) -> None:
    """Show distribution for a single die over many simulations."""
    rolls = [roll_die(sides) for _ in range(n_rolls)]
    counts = Counter(rolls)
    print(f"\n  d{sides} distribution over {n_rolls:,} rolls:")
    for face in range(1, sides + 1):
        p = counts[face] / n_rolls
        bar = "█" * int(p * sides * 20)
        print(f"  {face:>4}: {bar} {p:.3f}")


def interactive():
    print("=== Dice Roll Simulator ===")
    print("Enter dice notation like: 2d6  |  1d20+5  |  4d6 drop1  |  1d20 adv")
    print("Commands: roll <notation> | stats <sides> | history | quit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line: continue
        parts = line.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "history":
            if history:
                print("\n".join(history[-10:]))
            else:
                print("  No rolls yet.")
        elif cmd == "stats" and len(parts) >= 2:
            try:
                stats_display(int(parts[1]))
            except ValueError as e:
                print(f"  Error: {e}")
        else:
            # Treat entire input as roll notation with optional flags
            notation_part = parts[0] if parts[0][0].isdigit() or parts[0].startswith("d") else ""
            if not notation_part:
                print("  Usage: 2d6  or  1d20+5  or  4d6 drop1  or  1d20 adv")
                continue
            adv = "adv" in parts[1:]
            dis = "dis" in parts[1:] or "disadvantage" in parts[1:]
            drop = 0
            for p in parts[1:]:
                m = re.match(r"drop(\d+)", p)
                if m: drop = int(m.group(1))

            try:
                res = do_roll(notation_part, adv, dis, drop)
                display_result(res)
                entry = f"{notation_part} → {res['total']}"
                if res["label"]: entry += f" [{res['label']}]"
                history.append(entry)
            except ValueError as e:
                print(f"  Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Dice Roll Simulator")
    parser.add_argument("notation",  nargs="?", help="Dice notation, e.g. '2d6+3'")
    parser.add_argument("--adv",     action="store_true", help="Advantage (roll twice, take higher)")
    parser.add_argument("--dis",     action="store_true", help="Disadvantage (roll twice, take lower)")
    parser.add_argument("--drop",    type=int, default=0, metavar="N",
                        help="Drop N lowest dice (e.g. 4d6 --drop 1)")
    parser.add_argument("--stats",   type=int, metavar="SIDES",
                        help="Show distribution statistics for dSIDES")
    args = parser.parse_args()

    if args.stats:
        stats_display(args.stats)
    elif args.notation:
        try:
            res = do_roll(args.notation, args.adv, args.dis, args.drop)
            display_result(res)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
