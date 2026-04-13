"""Caesar Cipher Tool — CLI tool.

Encrypts and decrypts text using the Caesar cipher.
Also performs brute-force crack and frequency analysis.

Usage:
    python main.py
"""

import string
from collections import Counter


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def caesar_encrypt(text: str, shift: int) -> str:
    shift = shift % 26
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)


def caesar_decrypt(text: str, shift: int) -> str:
    return caesar_encrypt(text, -shift)


def brute_force(ciphertext: str) -> list[tuple[int, str]]:
    """Try all 25 shifts and return list of (shift, plaintext)."""
    return [(shift, caesar_decrypt(ciphertext, shift)) for shift in range(1, 26)]


def frequency_crack(ciphertext: str) -> tuple[int, str]:
    """Crack by comparing letter frequency to English 'e' (most common)."""
    letters_only = [c.lower() for c in ciphertext if c.isalpha()]
    if not letters_only:
        return 0, ciphertext

    counter = Counter(letters_only)
    most_common_cipher = counter.most_common(1)[0][0]
    # Most common letter in English is 'e'
    shift = (ord(most_common_cipher) - ord("e")) % 26
    return shift, caesar_decrypt(ciphertext, shift)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Caesar Cipher Tool
------------------
1. Encrypt text
2. Decrypt text (known shift)
3. Brute-force crack (try all shifts)
4. Frequency analysis crack
0. Quit
"""


def get_shift(prompt: str = "  Shift (1-25): ") -> int:
    while True:
        try:
            s = int(input(prompt).strip())
            if 1 <= s <= 25:
                return s
            print("  Shift must be between 1 and 25.")
        except ValueError:
            print("  Enter a number.")


def get_text(prompt: str) -> str:
    text = input(prompt).strip()
    return text


def main() -> None:
    print("Caesar Cipher Tool")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            text = get_text("  Plaintext: ")
            if not text:
                continue
            shift = get_shift()
            result = caesar_encrypt(text, shift)
            print(f"\n  Shift     : {shift}")
            print(f"  Encrypted : {result}")

        elif choice == "2":
            text = get_text("  Ciphertext: ")
            if not text:
                continue
            shift = get_shift("  Shift used for encryption: ")
            result = caesar_decrypt(text, shift)
            print(f"\n  Shift     : {shift}")
            print(f"  Decrypted : {result}")

        elif choice == "3":
            text = get_text("  Ciphertext: ")
            if not text:
                continue
            results = brute_force(text)
            print(f"\n  All {len(results)} shifts:")
            for shift, plain in results:
                print(f"  Shift {shift:>2}: {plain[:80]}")

        elif choice == "4":
            text = get_text("  Ciphertext: ")
            if not text:
                continue
            shift, plain = frequency_crack(text)
            print(f"\n  Best guess — shift: {shift}")
            print(f"  Decrypted: {plain}")
            print(f"  (Based on assuming 'e' is the most frequent letter)")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
