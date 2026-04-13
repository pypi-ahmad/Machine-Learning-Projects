"""Password Strength Checker — CLI tool.

Evaluates password strength against multiple criteria:
  - Length
  - Uppercase / lowercase / digit / special character presence
  - Common password check
  - Entropy estimation

Usage:
    python main.py
"""

import re
import math
import string
import getpass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMMON_PASSWORDS = {
    "password", "123456", "123456789", "qwerty", "abc123",
    "letmein", "monkey", "master", "dragon", "pass",
    "password1", "iloveyou", "sunshine", "princess", "welcome",
    "shadow", "superman", "michael", "login", "admin",
    "000000", "111111", "654321", "12345678", "1234567890",
}

MIN_LENGTHS = {
    "weak":   8,
    "fair":   10,
    "strong": 12,
    "great":  16,
}


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def estimate_entropy(password: str) -> float:
    """Estimate bits of entropy based on charset size and length."""
    charset = 0
    if any(c.islower() for c in password):
        charset += 26
    if any(c.isupper() for c in password):
        charset += 26
    if any(c.isdigit() for c in password):
        charset += 10
    if any(c in string.punctuation for c in password):
        charset += 32
    if charset == 0:
        return 0.0
    return len(password) * math.log2(charset)


def check_strength(password: str) -> dict:
    checks = {
        "length_8":       len(password) >= 8,
        "length_12":      len(password) >= 12,
        "length_16":      len(password) >= 16,
        "has_upper":      bool(re.search(r"[A-Z]", password)),
        "has_lower":      bool(re.search(r"[a-z]", password)),
        "has_digit":      bool(re.search(r"\d", password)),
        "has_special":    bool(re.search(r"[^A-Za-z0-9]", password)),
        "no_spaces":      " " not in password,
        "not_common":     password.lower() not in COMMON_PASSWORDS,
        "no_repeat":      not bool(re.search(r"(.)\1{2,}", password)),  # no 3+ same chars
        "no_sequential":  not _has_sequence(password),
    }

    score = sum(checks.values())
    entropy = estimate_entropy(password)

    if not checks["not_common"] or score <= 4:
        label = "Very Weak"
    elif score <= 6:
        label = "Weak"
    elif score <= 8:
        label = "Fair"
    elif score <= 10:
        label = "Strong"
    else:
        label = "Very Strong"

    return {
        "password_length": len(password),
        "checks":          checks,
        "score":           score,
        "max_score":       len(checks),
        "label":           label,
        "entropy_bits":    entropy,
    }


def _has_sequence(password: str) -> bool:
    """Detect common sequential patterns like 123, abc, qwerty."""
    lower = password.lower()
    sequences = ["0123456789", "abcdefghijklmnopqrstuvwxyz", "qwertyuiop", "asdfghjkl"]
    for seq in sequences:
        for i in range(len(seq) - 2):
            chunk = seq[i:i + 3]
            if chunk in lower or chunk[::-1] in lower:
                return True
    return False


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_STRENGTH_COLORS = {
    "Very Weak":   "✗✗✗✗✗",
    "Weak":        "✓✗✗✗✗",
    "Fair":        "✓✓✗✗✗",
    "Strong":      "✓✓✓✗✗",
    "Very Strong": "✓✓✓✓✓",
}

_CHECK_LABELS = {
    "length_8":      "At least 8 characters",
    "length_12":     "At least 12 characters",
    "length_16":     "At least 16 characters",
    "has_upper":     "Contains uppercase letter",
    "has_lower":     "Contains lowercase letter",
    "has_digit":     "Contains digit",
    "has_special":   "Contains special character",
    "no_spaces":     "No spaces",
    "not_common":    "Not a common password",
    "no_repeat":     "No 3+ repeated characters",
    "no_sequential": "No simple sequences (123, abc)",
}


def display_result(result: dict) -> None:
    label = result["label"]
    bar = _STRENGTH_COLORS.get(label, "")
    print(f"\n  Strength    : {label}  {bar}")
    print(f"  Score       : {result['score']}/{result['max_score']}")
    print(f"  Length      : {result['password_length']} characters")
    print(f"  Entropy     : {result['entropy_bits']:.1f} bits")
    print(f"\n  Checks:")
    for key, passed in result["checks"].items():
        icon = "  ✓" if passed else "  ✗"
        print(f"    {icon}  {_CHECK_LABELS.get(key, key)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("Password Strength Checker")
    print("=" * 40)

    while True:
        print("\n1. Check a password")
        print("2. Check multiple passwords (batch)")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            # Use getpass to hide input
            try:
                pwd = getpass.getpass("  Enter password (hidden): ")
            except Exception:
                pwd = input("  Enter password: ")
            if not pwd:
                print("  No password entered.")
                continue
            result = check_strength(pwd)
            display_result(result)

        elif choice == "2":
            print("  Enter passwords one per line (blank line to stop):")
            while True:
                try:
                    pwd = getpass.getpass("  > ")
                except Exception:
                    pwd = input("  > ")
                if not pwd:
                    break
                r = check_strength(pwd)
                bar = _STRENGTH_COLORS.get(r["label"], "")
                print(f"    → {r['label']:<12} {bar}  (score {r['score']}/{r['max_score']}, {r['entropy_bits']:.0f} bits)")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
