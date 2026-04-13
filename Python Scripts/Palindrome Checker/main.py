"""Palindrome Checker — CLI tool.

Checks whether a string, number, or sentence is a palindrome.
Handles words, sentences (ignoring spaces and punctuation),
and numbers. Also finds all palindromic substrings.

Usage:
    python main.py
"""

import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Strip non-alphanumeric characters and lowercase for comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def is_palindrome(text: str, strict: bool = False) -> bool:
    """Check if text is a palindrome.

    strict=True: compare as-is (no normalization).
    strict=False: ignore case, spaces, punctuation.
    """
    if strict:
        s = text
    else:
        s = normalize(text)
    return s == s[::-1]


def palindrome_substrings(text: str, min_length: int = 2) -> list[str]:
    """Find all unique palindromic substrings of length >= min_length."""
    results = set()
    n = len(text)
    for i in range(n):
        # Odd-length
        for j in range(0, min(i + 1, n - i)):
            sub = text[i - j:i + j + 1]
            if len(sub) >= min_length and sub == sub[::-1]:
                results.add(sub)
            else:
                if j > 0:
                    break
        # Even-length
        for j in range(0, min(i + 1, n - i - 1) + 1):
            sub = text[i - j:i + j + 2]
            if not sub:
                break
            if len(sub) >= min_length and sub == sub[::-1]:
                results.add(sub)
            else:
                if j > 0:
                    break
    return sorted(results, key=len, reverse=True)


def longest_palindrome(text: str) -> str:
    """Find the longest palindromic substring (Manacher-inspired)."""
    if not text:
        return ""
    s = text.lower()
    best = s[0]
    for i in range(len(s)):
        for j in (0, 1):
            lo, hi = i, i + j
            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                if hi - lo + 1 > len(best):
                    best = s[lo:hi + 1]
                lo -= 1
                hi += 1
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Palindrome Checker
------------------
1. Check if input is a palindrome
2. Find all palindromic substrings
3. Batch check (multiple inputs)
0. Quit
"""


def main() -> None:
    print("Palindrome Checker")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            text = input("  Enter text: ").strip()
            if not text:
                print("  No input.")
                continue
            normalized = normalize(text)
            result = is_palindrome(text)
            longest = longest_palindrome(normalized) if not result else normalized
            print(f"\n  Input (normalized): '{normalized}'")
            if result:
                print(f"  ✓ '{text}' IS a palindrome!")
            else:
                print(f"  ✗ '{text}' is NOT a palindrome.")
                print(f"  Closest palindrome fragment: '{longest}'")

        elif choice == "2":
            text = input("  Enter text: ").strip()
            if not text:
                print("  No input.")
                continue
            min_len_str = input("  Minimum substring length (default 2): ").strip()
            min_len = int(min_len_str) if min_len_str.isdigit() else 2
            subs = palindrome_substrings(text.lower(), min_len)
            if subs:
                print(f"\n  Found {len(subs)} palindromic substring(s):")
                for s in subs[:20]:
                    print(f"    '{s}' (length {len(s)})")
                if len(subs) > 20:
                    print(f"    ... and {len(subs) - 20} more")
            else:
                print("  No palindromic substrings found.")

        elif choice == "3":
            print("  Enter inputs one per line (blank line to stop):")
            while True:
                text = input("  > ").strip()
                if not text:
                    break
                result = is_palindrome(text)
                icon = "✓" if result else "✗"
                print(f"    {icon}  '{text}'")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
