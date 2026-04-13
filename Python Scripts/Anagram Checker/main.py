"""Anagram Checker — CLI tool.

Checks if two strings are anagrams, finds all anagrams of a word
from a wordlist, and generates anagram groups from a list of words.

Usage:
    python main.py
"""

from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def normalize_word(word: str) -> str:
    return "".join(sorted(word.lower().replace(" ", "")))


def are_anagrams(a: str, b: str) -> bool:
    """Check if two strings are anagrams of each other."""
    return normalize_word(a) == normalize_word(b)


def find_anagrams_in_list(word: str, word_list: list[str]) -> list[str]:
    """Find all anagrams of 'word' in 'word_list'."""
    key = normalize_word(word)
    return [w for w in word_list if normalize_word(w) == key and w.lower() != word.lower()]


def group_anagrams(words: list[str]) -> list[list[str]]:
    """Group a list of words by anagram family."""
    groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        groups[normalize_word(word)].append(word)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def load_wordlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def main() -> None:
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
                print(f"\n  ✓ '{a}' and '{b}' ARE anagrams!")
            else:
                print(f"\n  ✗ '{a}' and '{b}' are NOT anagrams.")
                # Show diff
                ca = Counter(normalize_word(a))
                cb = Counter(normalize_word(b))
                only_a = ca - cb
                only_b = cb - ca
                if only_a:
                    print(f"  Extra in '{a}': {dict(only_a)}")
                if only_b:
                    print(f"  Extra in '{b}': {dict(only_b)}")

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


if __name__ == "__main__":
    main()
