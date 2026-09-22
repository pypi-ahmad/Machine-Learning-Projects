"""Check whether a word satisfies the Letter Partners nesting rules."""

from __future__ import annotations

import argparse

PRE_PARTNERS = "abcdefghijklm"
POST_PARTNERS = "nopqrstuvwxyz"


def evaluate_word(word: str) -> tuple[bool, str]:
    """Return whether a word has valid partner ordering and nesting."""
    normalized = word.strip().lower()
    if not normalized:
        return False, "Enter at least one letter."
    if not normalized.isalpha():
        return False, "Use letters only."

    expected_closers: list[str] = []
    for letter in normalized:
        if letter in PRE_PARTNERS:
            expected_closers.append(POST_PARTNERS[PRE_PARTNERS.index(letter)])
        elif letter in expected_closers:
            if letter != expected_closers[-1]:
                return False, f"{letter!r} closes a partner before its nested partner."
            expected_closers.pop()

    if expected_closers:
        return False, f"Missing partner {expected_closers[-1]!r}."
    return True, "All pre-partners are correctly ordered and nested."


def parse_args() -> argparse.Namespace:
    """Parse an optional word to evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word", nargs="?", help="Word containing Letter Partners")
    return parser.parse_args()


def main() -> None:
    """Read and evaluate a word."""
    args = parse_args()
    word = args.word or input("Enter a word: ")
    won, reason = evaluate_word(word)
    print("GAME WON" if won else "GAME LOST")
    print(reason)


if __name__ == "__main__":
    main()
