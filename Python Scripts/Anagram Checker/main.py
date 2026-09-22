"""Anagram Checker - CLI tool.

Checks if two strings are anagrams, finds all anagrams of a word
from a wordlist, and generates anagram groups from a list of words.

Usage:
    python main.py
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def normalize_word(word: str) -> str:
    """Return a case-insensitive key that ignores whitespace."""
    return "".join(sorted(character for character in word.casefold() if not character.isspace()))


def are_anagrams(a: str, b: str) -> bool:
    """Check if two strings are anagrams of each other."""
    return normalize_word(a) == normalize_word(b)


def find_anagrams_in_list(word: str, word_list: list[str]) -> list[str]:
    """Find all anagrams of 'word' in 'word_list'."""
    key = normalize_word(word)
    return [w for w in word_list if normalize_word(w) == key and w.casefold() != word.casefold()]


def group_anagrams(words: list[str]) -> list[list[str]]:
    """Group a list of words by anagram family."""
    groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        groups[normalize_word(word)].append(word)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def load_wordlist(path: Path) -> list[str]:
    """Load non-empty wordlist entries from a UTF-8 text file."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Anagram Checker
---------------
1. Check if two words/phrases are anagrams
2. Group a list of words by anagram family
3. Find anagrams from a wordlist file
0. Quit
"""


def print_check(first: str, second: str) -> None:
    """Print an anagram result and any unmatched characters."""
    if are_anagrams(first, second):
        print(f"{first!r} and {second!r} are anagrams.")
        return

    print(f"{first!r} and {second!r} are not anagrams.")
    first_characters = Counter(normalize_word(first))
    second_characters = Counter(normalize_word(second))
    only_first = first_characters - second_characters
    only_second = second_characters - first_characters
    if only_first:
        print(f"Extra in {first!r}: {dict(only_first)}")
    if only_second:
        print(f"Extra in {second!r}: {dict(only_second)}")


def interactive_mode() -> None:
    print("Anagram Checker")

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            a = input("  First word/phrase: ").strip()
            b = input("  Second word/phrase: ").strip()
            if not a or not b:
                print("  Both inputs required.")
                continue
            result = are_anagrams(a, b)
            if result:
                print(f"\n  {a!r} and {b!r} are anagrams.")
            else:
                print()
                print_check(a, b)

        elif choice == "2":
            print("  Enter words one per line (blank line to finish):")
            words = []
            while True:
                w = input("  > ").strip()
                if not w:
                    break
                words.extend(w.split())  # allow space-separated too

            if len(words) < 2:
                print("  Need at least 2 words.")
                continue

            groups = group_anagrams(words)
            if groups:
                print(f"\n  Found {len(groups)} anagram group(s):")
                for i, group in enumerate(groups, 1):
                    print(f"    {i}. {' | '.join(group)}")
            else:
                print("\n  No anagram groups found.")

        elif choice == "3":
            file_path = input("  Path to wordlist file: ").strip().strip('"')
            wordlist = load_wordlist(Path(file_path))
            if not wordlist:
                print(f"  Could not load wordlist from: {file_path}")
                continue
            word = input(f"  Word to search (in {len(wordlist):,} words): ").strip()
            if not word:
                continue
            results = find_anagrams_in_list(word, wordlist)
            if results:
                print(f"\n  Anagrams of '{word}':")
                for r in results[:50]:
                    print(f"    {r}")
                if len(results) > 50:
                    print(f"    ... and {len(results) - 50} more")
            else:
                print(f"\n  No anagrams of '{word}' found in the wordlist.")

        else:
            print("  Invalid choice.")


def build_parser() -> argparse.ArgumentParser:
    """Build the non-interactive command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    check = commands.add_parser("check", help="check two words or phrases")
    check.add_argument("first")
    check.add_argument("second")

    group = commands.add_parser("group", help="group words by anagram family")
    group.add_argument("words", nargs="+", help="two or more words")

    find = commands.add_parser("find", help="find a word's anagrams in a file")
    find.add_argument("word")
    find.add_argument("wordlist", type=Path)
    return parser


def main() -> None:
    """Run a selected command or open the interactive menu."""
    args = build_parser().parse_args()
    if args.command is None:
        interactive_mode()
    elif args.command == "check":
        print_check(args.first, args.second)
    elif args.command == "group":
        groups = group_anagrams(args.words)
        if groups:
            for group in groups:
                print(" | ".join(group))
        else:
            print("No anagram groups found.")
    elif args.command == "find":
        try:
            results = find_anagrams_in_list(args.word, load_wordlist(args.wordlist))
        except OSError as error:
            raise SystemExit(f"Error: {error}") from error
        if results:
            print("\n".join(results))
        else:
            print("No anagrams found.")


if __name__ == "__main__":
    main()
