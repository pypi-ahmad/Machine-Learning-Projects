"""Emoji Guessing Game — CLI game.

Guess the word or phrase from a sequence of emojis.
Multiple categories: movies, food, animals, phrases.

Usage:
    python main.py
    python main.py --category movies --rounds 10
"""

import argparse
import random
import time


PUZZLES = {
    "movies": [
        ("🦁👑",              "the lion king"),
        ("🕷️🕸️👨",           "spider man"),
        ("🧊❄️🏔️",            "frozen"),
        ("🦈🎬",              "jaws"),
        ("👻🏠",              "ghost"),
        ("🚀🌌⭐",            "star wars"),
        ("🧙‍♂️💍🌋",           "lord of the rings"),
        ("🤖🚗",              "transformers"),
        ("🦸‍♂️🕵️‍♂️",          "batman"),
        ("🧜‍♀️🌊",             "the little mermaid"),
        ("🍕🐢🥷",            "teenage mutant ninja turtles"),
        ("👔💰🏦",            "the wolf of wall street"),
        ("🚢💔🌊",            "titanic"),
        ("🦋🔫",              "silence of the lambs"),
        ("🏈🍫🌲",            "forrest gump"),
    ],
    "food": [
        ("🍞🫙🥜",            "peanut butter sandwich"),
        ("🍕🧀🍅",            "pizza margherita"),
        ("🥑🍞",              "avocado toast"),
        ("🍝🥩🧅",            "spaghetti bolognese"),
        ("🌮🌶️🧅",            "taco"),
        ("🥞🍯🫙",            "pancakes with honey"),
        ("🍜🍗🥕",            "chicken noodle soup"),
        ("🍦🍫🍓",            "chocolate strawberry ice cream"),
        ("🥗🥦🥕🫑",          "vegetable salad"),
        ("🍰🎂🎉",            "birthday cake"),
    ],
    "animals": [
        ("🌊🐻",              "polar bear"),
        ("🦓🐴",              "zebra"),
        ("🌿🐘",              "elephant"),
        ("🌙🦇",              "bat"),
        ("🌊🐟",              "shark"),
        ("🌲🐿️",              "squirrel"),
        ("🔥🐦",              "phoenix"),
        ("🌸🦢",              "swan"),
        ("🧊🐧",              "penguin"),
        ("🌙🦉",              "owl"),
    ],
    "phrases": [
        ("🌧️🏃",              "running in the rain"),
        ("🔑❤️",              "key to my heart"),
        ("🌍🕊️",              "world peace"),
        ("🕐🏖️",              "vacation time"),
        ("💡⏰",              "light bulb moment"),
        ("🎵🌧️",              "singing in the rain"),
        ("🏃💨",              "running on empty"),
        ("⬆️📈",              "up and coming"),
        ("🐠🎣",              "fishing for compliments"),
        ("🌊🏄",              "surfing the wave"),
    ],
}


def play_round(puzzle: tuple, num: int, total: int, hints_enabled: bool) -> bool:
    emojis, answer = puzzle
    words   = answer.split()
    print(f"\n  Round {num}/{total}")
    print(f"  🔸 {emojis}")
    print(f"  ({len(words)} word{'s' if len(words)>1 else ''})", end="")
    print(f"  Letters: {', '.join(str(len(w)) for w in words)}")

    hint_used = False
    attempt   = 0
    max_attempts = 3

    while attempt < max_attempts:
        try:
            ans = input(f"  Your guess (attempt {attempt+1}/{max_attempts}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if ans == answer:
            bonus = 3 - attempt
            print(f"  ✅ Correct! {'(first try! +bonus)' if attempt==0 else ''}")
            return True
        elif ans in ("hint", "h") and hints_enabled and not hint_used:
            revealed = " ".join(w[0] + "_" * (len(w)-1) for w in words)
            print(f"  Hint: {revealed}")
            hint_used = True
        elif ans in ("skip", "s"):
            print(f"  Skipped. Answer: {answer}")
            return False
        else:
            attempt += 1
            if attempt < max_attempts:
                # Partial match hint
                user_words = ans.split()
                matches    = sum(1 for uw, aw in zip(user_words, words) if uw == aw)
                if matches > 0:
                    print(f"  ❌ Close! {matches} word(s) correct.")
                else:
                    print(f"  ❌ Try again.")

    print(f"  ❌ Out of attempts. Answer: {answer}")
    return False


def play(category: str, n_rounds: int, hints: bool) -> None:
    pool  = PUZZLES.get(category) or [p for ps in PUZZLES.values() for p in ps]
    puzzles = random.sample(pool, min(n_rounds, len(pool)))
    score   = 0

    print(f"\n=== Emoji Guessing Game ===  [{category.title()}]")
    print(f"  Rounds: {n_rounds}  |  Hints: {'on' if hints else 'off'}")
    print("  Type 'hint' for a letter hint, 'skip' to skip\n")

    for i, puzzle in enumerate(puzzles, 1):
        if play_round(puzzle, i, len(puzzles), hints):
            score += 1

    pct = score / len(puzzles) * 100
    print(f"\n  ── Results ──")
    print(f"  Correct: {score}/{len(puzzles)}")
    print(f"  Accuracy: {pct:.0f}%")
    if pct == 100: print("  🏆 Perfect!")
    elif pct >= 60: print("  👍 Good job!")
    else: print("  🤔 Keep practicing!")


def main():
    cats   = list(PUZZLES.keys()) + ["all"]
    parser = argparse.ArgumentParser(description="Emoji Guessing Game")
    parser.add_argument("--category", choices=cats, default=None)
    parser.add_argument("--rounds",   type=int, default=5)
    parser.add_argument("--no-hints", action="store_true")
    args = parser.parse_args()

    if not args.category:
        print("=== Emoji Guessing Game ===")
        print(f"Categories: {', '.join(cats)}")
        cat = input("Choose category: ").strip().lower()
        if cat not in cats: cat = "all"
        args.category = cat

    play(args.category, args.rounds, not args.no_hints)


if __name__ == "__main__":
    main()
