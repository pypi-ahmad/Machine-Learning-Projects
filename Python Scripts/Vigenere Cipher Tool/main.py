"""Vigenère Cipher Tool — CLI tool.

Encrypts and decrypts text using the Vigenère cipher.
Also performs Kasiski / Index-of-Coincidence key-length estimation
and frequency-analysis-based key recovery.

Usage:
    python main.py
"""

import string
from collections import Counter
from itertools import cycle


# ---------------------------------------------------------------------------
# Core cipher
# ---------------------------------------------------------------------------

def vigenere_encrypt(text: str, key: str) -> str:
    """Encrypt plaintext with a Vigenère key (letters only, preserves case)."""
    key_clean = [c.lower() for c in key if c.isalpha()]
    if not key_clean:
        raise ValueError("Key must contain at least one letter.")
    result = []
    key_iter = cycle(key_clean)
    for ch in text:
        if ch.isalpha():
            shift = ord(next(key_iter)) - ord("a")
            base  = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)


def vigenere_decrypt(text: str, key: str) -> str:
    """Decrypt ciphertext with a Vigenère key."""
    key_clean = [c.lower() for c in key if c.isalpha()]
    if not key_clean:
        raise ValueError("Key must contain at least one letter.")
    result = []
    key_iter = cycle(key_clean)
    for ch in text:
        if ch.isalpha():
            shift = ord(next(key_iter)) - ord("a")
            base  = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base - shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)


# ---------------------------------------------------------------------------
# Cryptanalysis helpers
# ---------------------------------------------------------------------------

ENGLISH_FREQ = {
    "e": 12.70, "t": 9.06, "a": 8.17, "o": 7.51, "i": 6.97, "n": 6.75,
    "s": 6.33, "h": 6.09, "r": 5.99, "d": 4.25, "l": 4.03, "c": 2.78,
    "u": 2.76, "m": 2.41, "w": 2.36, "f": 2.23, "g": 2.02, "y": 1.97,
    "p": 1.93, "b": 1.49, "v": 0.98, "k": 0.77, "j": 0.15, "x": 0.15,
    "q": 0.10, "z": 0.07,
}
EXPECTED_IC = 0.0667  # English


def index_of_coincidence(letters: str) -> float:
    n = len(letters)
    if n < 2:
        return 0.0
    counts = Counter(letters)
    numerator = sum(c * (c - 1) for c in counts.values())
    return numerator / (n * (n - 1))


def estimate_key_length(ciphertext: str, max_kl: int = 20) -> list[tuple[int, float]]:
    """Return (key_length, avg_IC) sorted by closeness to English IC."""
    letters = [c.lower() for c in ciphertext if c.isalpha()]
    results = []
    for kl in range(2, min(max_kl + 1, len(letters) // 2)):
        ics = []
        for offset in range(kl):
            sub = "".join(letters[offset::kl])
            ics.append(index_of_coincidence(sub))
        avg_ic = sum(ics) / len(ics) if ics else 0
        results.append((kl, avg_ic))
    results.sort(key=lambda x: abs(x[1] - EXPECTED_IC))
    return results[:5]


def guess_shift_for_stream(letters: str) -> int:
    """Guess the Caesar shift for a single Vigenère column by frequency chi-sq."""
    n = len(letters)
    if n == 0:
        return 0
    counts = Counter(letters)
    best_shift, best_score = 0, float("inf")
    for shift in range(26):
        score = 0.0
        for i, ch in enumerate(string.ascii_lowercase):
            observed = counts.get(chr((i + shift) % 26 + ord("a")), 0)
            expected  = ENGLISH_FREQ[ch] / 100 * n
            if expected:
                score += (observed - expected) ** 2 / expected
        if score < best_score:
            best_score = score
            best_shift = shift
    return best_shift


def crack_key(ciphertext: str, key_length: int) -> str:
    """Recover the key using frequency analysis column-by-column."""
    letters = [c.lower() for c in ciphertext if c.isalpha()]
    key = []
    for offset in range(key_length):
        column = "".join(letters[offset::key_length])
        shift  = guess_shift_for_stream(column)
        key.append(chr(shift + ord("a")))
    return "".join(key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Vigenère Cipher Tool
--------------------
1. Encrypt text
2. Decrypt text (known key)
3. Estimate key length (IC analysis)
4. Full crack (estimate length + recover key)
0. Quit
"""


def get_key(prompt: str = "  Key: ") -> str:
    while True:
        key = input(prompt).strip()
        if key and all(c.isalpha() or c.isspace() for c in key):
            return key.replace(" ", "")
        print("  Key must contain only letters.")


def get_text(prompt: str) -> str:
    return input(prompt).strip()


def main() -> None:
    print("Vigenère Cipher Tool")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            text = get_text("  Plaintext : ")
            if not text:
                continue
            key = get_key()
            try:
                result = vigenere_encrypt(text, key)
                print(f"\n  Key       : {key}")
                print(f"  Encrypted : {result}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "2":
            text = get_text("  Ciphertext: ")
            if not text:
                continue
            key = get_key()
            try:
                result = vigenere_decrypt(text, key)
                print(f"\n  Key       : {key}")
                print(f"  Decrypted : {result}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "3":
            text = get_text("  Ciphertext: ")
            if not text:
                continue
            estimates = estimate_key_length(text)
            print("\n  Top key-length estimates (closest IC to English):")
            print(f"  {'Length':>8}  {'Avg IC':>8}  {'Δ from 0.0667':>14}")
            for kl, ic in estimates:
                print(f"  {kl:>8}  {ic:>8.4f}  {abs(ic - EXPECTED_IC):>14.4f}")

        elif choice == "4":
            text = get_text("  Ciphertext: ")
            if not text:
                continue
            estimates = estimate_key_length(text)
            best_kl   = estimates[0][0]
            print(f"\n  Best estimated key length: {best_kl}")
            key = crack_key(text, best_kl)
            print(f"  Recovered key            : {key}")
            try:
                plain = vigenere_decrypt(text, key)
                print(f"  Decrypted (first 200):   {plain[:200]}")
            except ValueError:
                pass

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
