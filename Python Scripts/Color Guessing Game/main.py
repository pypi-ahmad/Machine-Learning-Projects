"""Color Guessing Game — CLI game.

Guess the CSS hex color from RGB values or name clues.
Multiple game modes: hex-to-name, RGB-to-hex, color mixing.

Usage:
    python main.py
    python main.py --mode hex
    python main.py --rounds 10
"""

import argparse
import random


# (name, hex, r, g, b)
COLORS = [
    ("red",         "#FF0000", 255,   0,   0),
    ("green",       "#008000",   0, 128,   0),
    ("blue",        "#0000FF",   0,   0, 255),
    ("yellow",      "#FFFF00", 255, 255,   0),
    ("cyan",        "#00FFFF",   0, 255, 255),
    ("magenta",     "#FF00FF", 255,   0, 255),
    ("orange",      "#FFA500", 255, 165,   0),
    ("purple",      "#800080", 128,   0, 128),
    ("pink",        "#FFC0CB", 255, 192, 203),
    ("brown",       "#A52A2A", 165,  42,  42),
    ("white",       "#FFFFFF", 255, 255, 255),
    ("black",       "#000000",   0,   0,   0),
    ("gray",        "#808080", 128, 128, 128),
    ("lime",        "#00FF00",   0, 255,   0),
    ("navy",        "#000080",   0,   0, 128),
    ("teal",        "#008080",   0, 128, 128),
    ("maroon",      "#800000", 128,   0,   0),
    ("olive",       "#808000", 128, 128,   0),
    ("coral",       "#FF7F50", 255, 127,  80),
    ("salmon",      "#FA8072", 250, 128, 114),
    ("turquoise",   "#40E0D0",  64, 224, 208),
    ("indigo",      "#4B0082",  75,   0, 130),
    ("violet",      "#EE82EE", 238, 130, 238),
    ("gold",        "#FFD700", 255, 215,   0),
    ("silver",      "#C0C0C0", 192, 192, 192),
    ("crimson",     "#DC143C", 220,  20,  60),
    ("khaki",       "#F0E68C", 240, 230, 140),
    ("lavender",    "#E6E6FA", 230, 230, 250),
    ("ivory",       "#FFFFF0", 255, 255, 240),
    ("beige",       "#F5F5DC", 245, 245, 220),
]


def color_bar(r: int, g: int, b: int, width: int = 20) -> str:
    """ANSI colored bar if terminal supports it."""
    block = "█" * width
    return f"\033[38;2;{r};{g};{b}m{block}\033[0m"


def hex_distance(c1, c2) -> float:
    return ((c1[2]-c2[2])**2 + (c1[3]-c2[3])**2 + (c1[4]-c2[4])**2) ** 0.5


# ── Game modes ────────────────────────────────────────────────────────────────

def mode_name_to_hex(n_rounds: int) -> None:
    """Show color name → guess the hex code."""
    score = 0
    print("\n  Mode: Name → Hex Code\n")
    pool = random.sample(COLORS, min(n_rounds, len(COLORS)))

    for i, color in enumerate(pool, 1):
        name, correct_hex, r, g, b = color
        print(f"  Round {i}/{len(pool)}: What is the hex code for '{name}'?")
        print(f"  {color_bar(r, g, b)}")
        try:
            ans = input("  # ").strip().upper().lstrip("#")
        except (EOFError, KeyboardInterrupt):
            break
        if ans == correct_hex.lstrip("#").upper():
            print(f"  ✅ Correct! #{ans}")
            score += 10
        else:
            print(f"  ❌ Wrong. Correct: {correct_hex}  (RGB: {r},{g},{b})")
        print()

    print(f"  Score: {score}/{len(pool)*10}")


def mode_rgb_to_name(n_rounds: int) -> None:
    """Show RGB values → guess the color name from 4 options."""
    score = 0
    print("\n  Mode: RGB Values → Color Name\n")
    pool  = random.sample(COLORS, min(n_rounds, len(COLORS)))

    for i, color in enumerate(pool, 1):
        name, hex_, r, g, b = color
        # Build 3 wrong choices (closest colors for plausibility)
        others = sorted([c for c in COLORS if c[0] != name],
                        key=lambda c: hex_distance(color, c))
        wrong  = [c[0] for c in random.sample(others[:8], 3)]
        options = [name] + wrong
        random.shuffle(options)
        correct_idx = options.index(name) + 1

        print(f"  Round {i}/{len(pool)}: RGB({r}, {g}, {b})")
        print(f"  {color_bar(r, g, b)}")
        for j, opt in enumerate(options, 1):
            print(f"    {j}. {opt}")

        try:
            ans = input("  Your answer (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if ans.isdigit() and int(ans) == correct_idx:
            print(f"  ✅ Correct! It's {name}")
            score += 10
        else:
            print(f"  ❌ Wrong. The color was: {name}")
        print()

    print(f"  Score: {score}/{len(pool)*10}")


def mode_hex_to_name(n_rounds: int) -> None:
    """Show hex code → guess the color name from 4 options."""
    score = 0
    print("\n  Mode: Hex Code → Color Name\n")
    pool  = random.sample(COLORS, min(n_rounds, len(COLORS)))

    for i, color in enumerate(pool, 1):
        name, hex_, r, g, b = color
        others = sorted([c for c in COLORS if c[0] != name],
                        key=lambda c: hex_distance(color, c))
        wrong  = [c[0] for c in random.sample(others[:8], 3)]
        options = [name] + wrong
        random.shuffle(options)
        correct_idx = options.index(name) + 1

        print(f"  Round {i}/{len(pool)}: {hex_}")
        print(f"  {color_bar(r, g, b)}")
        for j, opt in enumerate(options, 1):
            print(f"    {j}. {opt}")

        try:
            ans = input("  Your answer (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if ans.isdigit() and int(ans) == correct_idx:
            print(f"  ✅ Correct!")
            score += 10
        else:
            print(f"  ❌ Wrong. The color was: {name}")
        print()

    print(f"  Score: {score}/{len(pool)*10}")


def interactive(n_rounds: int) -> None:
    print("=== Color Guessing Game ===")
    print("Modes:")
    print("  1. RGB → Name  (see RGB values, pick the name)")
    print("  2. Hex → Name  (see hex code, pick the name)")
    print("  3. Name → Hex  (see name, type the hex code)\n")
    choice = input("Choose mode (1/2/3): ").strip()
    if   choice == "1": mode_rgb_to_name(n_rounds)
    elif choice == "2": mode_hex_to_name(n_rounds)
    elif choice == "3": mode_name_to_hex(n_rounds)
    else: print("Invalid. Defaulting to mode 1."); mode_rgb_to_name(n_rounds)


def main():
    modes = {"rgb": mode_rgb_to_name, "hex": mode_hex_to_name, "name": mode_name_to_hex}
    parser = argparse.ArgumentParser(description="Color Guessing Game")
    parser.add_argument("--mode",   choices=list(modes.keys()), default=None)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if args.mode:
        modes[args.mode](args.rounds)
    else:
        interactive(args.rounds)


if __name__ == "__main__":
    main()
