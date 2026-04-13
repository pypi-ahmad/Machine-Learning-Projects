"""Wordle Clone — CLI game.

Guess the 5-letter word in 6 tries.
Color-coded feedback: green = right position, yellow = wrong position,
grey = not in word.  Includes a built-in word list and hard mode.

Usage:
    python main.py
"""

import random
import string


# ---------------------------------------------------------------------------
# Word list (common 5-letter words)
# ---------------------------------------------------------------------------

WORDS = [
    "apple", "brave", "crane", "drift", "eagle", "flame", "grace", "haven",
    "ivory", "joust", "knave", "lance", "maple", "noble", "opera", "piano",
    "quest", "raven", "shirt", "tiger", "ultra", "viper", "witch", "xenon",
    "yacht", "zonal", "abode", "blaze", "chess", "daisy", "elite", "flock",
    "glide", "hedge", "indie", "jazzy", "kneel", "lemon", "mirth", "nerve",
    "ocean", "plank", "queen", "rhino", "stale", "thorn", "unity", "viola",
    "wrist", "yield", "amber", "boxer", "choir", "dance", "ember", "frond",
    "gruel", "hippo", "inlet", "jelly", "kayak", "lymph", "magic", "nymph",
    "ozone", "prism", "quirk", "risky", "snore", "tribe", "usher", "vouch",
    "waltz", "extra", "yeast", "zebra", "abyss", "blunt", "crisp", "depth",
    "exact", "frown", "guild", "humor", "imply", "joker", "knife", "light",
    "midst", "ninth", "orbit", "plumb", "rapid", "stark", "thick", "umber",
    "vault", "whelp", "expel", "young", "zesty", "abbey", "blend", "cleft",
    "dwarf", "enjoy", "fjord", "glyph", "hatch", "input", "jumbo", "knock",
]

VALID_GUESSES = set(w.lower() for w in WORDS)

# ANSI colors
GREEN  = "\033[42m\033[30m"
YELLOW = "\033[43m\033[30m"
GREY   = "\033[100m\033[37m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------

def score_guess(guess: str, target: str) -> list[int]:
    """Return list of 0=grey, 1=yellow, 2=green."""
    result = [0] * 5
    target_list = list(target)
    # First pass: greens
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = 2
            target_list[i] = None
    # Second pass: yellows
    for i in range(5):
        if result[i] == 0 and guess[i] in target_list:
            result[i] = 1
            target_list[target_list.index(guess[i])] = None
    return result


def colorize_row(guess: str, scores: list[int]) -> str:
    parts = []
    for ch, s in zip(guess.upper(), scores):
        if s == 2:
            parts.append(f"{GREEN} {ch} {RESET}")
        elif s == 1:
            parts.append(f"{YELLOW} {ch} {RESET}")
        else:
            parts.append(f"{GREY} {ch} {RESET}")
    return " ".join(parts)


def display_keyboard(guesses: list[str], scores: list[list[int]]) -> None:
    """Show keyboard with colour hints."""
    letter_state: dict[str, int] = {}
    for guess, score in zip(guesses, scores):
        for ch, s in zip(guess, score):
            cur = letter_state.get(ch, -1)
            letter_state[ch] = max(cur, s)

    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    print()
    for row in rows:
        print("  ", end="")
        for ch in row:
            s = letter_state.get(ch, -1)
            if s == 2:
                print(f"{GREEN}{ch.upper()}{RESET}", end=" ")
            elif s == 1:
                print(f"{YELLOW}{ch.upper()}{RESET}", end=" ")
            elif s == 0:
                print(f"{GREY}{ch.upper()}{RESET}", end=" ")
            else:
                print(ch.upper(), end=" ")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def play_game(hard_mode: bool = False) -> None:
    target = random.choice(WORDS).lower()
    max_tries = 6
    guesses: list[str]     = []
    all_scores: list[list] = []

    print(f"\n  {'WORDLE CLONE':^35}")
    print(f"  Guess the 5-letter word in {max_tries} tries!")
    if hard_mode:
        print("  HARD MODE: must use revealed hints.")
    print()

    for attempt in range(1, max_tries + 1):
        # Print grid
        for i in range(max_tries):
            if i < len(guesses):
                print(f"  {attempt - len(guesses) + i - 1 + len(guesses):>2}  "
                      f"{colorize_row(guesses[i], all_scores[i])}")
            elif i == len(guesses):
                print(f"  {i+1:>2}  [ _ ] [ _ ] [ _ ] [ _ ] [ _ ]  ← current")
            else:
                print(f"  {i+1:>2}  [ _ ] [ _ ] [ _ ] [ _ ] [ _ ]")
        display_keyboard(guesses, all_scores)

        # Get guess
        while True:
            guess = input(f"\n  Attempt {attempt}/{max_tries}: ").strip().lower()
            if len(guess) != 5:
                print("  Must be 5 letters.")
                continue
            if not guess.isalpha():
                print("  Letters only.")
                continue
            # Hard mode: must use known greens
            if hard_mode and guesses:
                last_score = all_scores[-1]
                last_guess = guesses[-1]
                valid = True
                for i, (s, ch) in enumerate(zip(last_score, last_guess)):
                    if s == 2 and guess[i] != ch:
                        print(f"  Hard mode: position {i+1} must be '{ch.upper()}'.")
                        valid = False
                        break
                if not valid:
                    continue
            break

        scores = score_guess(guess, target)
        guesses.append(guess)
        all_scores.append(scores)

        # Clear and reprint
        if guess == target:
            print(f"\n  {colorize_row(guess, scores)}")
            praises = ["Genius!", "Magnificent!", "Impressive!", "Splendid!", "Great!", "Phew!"]
            print(f"\n  🎉 {praises[attempt - 1]}  Solved in {attempt} tries!")
            return

    print(f"\n  {colorize_row(target, [2]*5)}")
    print(f"\n  Game over!  The word was: {BOLD}{target.upper()}{RESET}")


MENU = """
Wordle Clone
------------
1. Play (normal mode)
2. Play (hard mode)
0. Quit
"""


def main() -> None:
    print("Wordle Clone")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()
        if choice == "0":
            print("Bye!")
            break
        elif choice == "1":
            play_game(hard_mode=False)
        elif choice == "2":
            play_game(hard_mode=True)
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
