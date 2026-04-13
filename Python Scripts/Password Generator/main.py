"""Password Generator — CLI tool.

Generate secure random passwords with configurable length,
character sets, and custom rules. Also checks strength.

Usage:
    python main.py
    python main.py 20
    python main.py 16 --no-symbols
"""

import argparse
import secrets
import string
import sys


LOWER   = string.ascii_lowercase
UPPER   = string.ascii_uppercase
DIGITS  = string.digits
SYMBOLS = "!@#$%^&*()-_=+[]{}|;:,.<>?"
SIMILAR = "il1Lo0O"  # visually confusing chars

WORD_LIST = [
    "apple","brave","cloud","dance","eagle","flame","grace","honey",
    "ivory","jungle","kneel","lemon","magic","noble","ocean","prism",
    "quest","river","stone","tiger","ultra","vivid","watch","xenon",
    "yacht","zebra","amber","blaze","crisp","dwell","ember","frost",
    "glare","haven","infer","joust","karma","lunar","mirth","night",
]


def strength(pwd: str) -> tuple[str, int]:
    score = 0
    if len(pwd) >= 8:  score += 1
    if len(pwd) >= 12: score += 1
    if len(pwd) >= 16: score += 1
    if any(c in LOWER  for c in pwd): score += 1
    if any(c in UPPER  for c in pwd): score += 1
    if any(c in DIGITS for c in pwd): score += 1
    if any(c in SYMBOLS for c in pwd): score += 1
    label = {0:"Very Weak",1:"Weak",2:"Weak",3:"Fair",
             4:"Good",5:"Strong",6:"Strong",7:"Very Strong"}.get(score,"Fair")
    return label, score


def generate(length: int = 16, use_upper: bool = True, use_digits: bool = True,
             use_symbols: bool = True, exclude_similar: bool = False) -> str:
    pool = LOWER
    required = [secrets.choice(LOWER)]
    if use_upper:
        pool += UPPER
        required.append(secrets.choice(UPPER))
    if use_digits:
        pool += DIGITS
        required.append(secrets.choice(DIGITS))
    if use_symbols:
        pool += SYMBOLS
        required.append(secrets.choice(SYMBOLS))
    if exclude_similar:
        pool = "".join(c for c in pool if c not in SIMILAR)

    extra = length - len(required)
    if extra < 0:
        extra = 0
    chars = required + [secrets.choice(pool) for _ in range(extra)]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars[:length])


def generate_passphrase(words: int = 4, separator: str = "-") -> str:
    chosen = [secrets.choice(WORD_LIST) for _ in range(words)]
    return separator.join(chosen) + separator + str(secrets.randbelow(999))


def main():
    parser = argparse.ArgumentParser(description="Password Generator")
    parser.add_argument("length", nargs="?", type=int, default=None)
    parser.add_argument("--no-symbols", action="store_true")
    parser.add_argument("--no-upper",   action="store_true")
    parser.add_argument("--no-digits",  action="store_true")
    parser.add_argument("--passphrase", action="store_true")
    parser.add_argument("--count",      type=int, default=1)
    args = parser.parse_args()

    if args.length is not None:
        # Quick single-shot mode
        for _ in range(args.count):
            if args.passphrase:
                pwd = generate_passphrase()
            else:
                pwd = generate(
                    length=args.length,
                    use_upper=not args.no_upper,
                    use_digits=not args.no_digits,
                    use_symbols=not args.no_symbols,
                )
            label, score = strength(pwd)
            print(f"  {pwd}   [{label}]")
        return

    # Interactive mode
    print("Password Generator")
    print("──────────────────────────────")

    while True:
        print("\n  1. Generate password")
        print("  2. Generate passphrase")
        print("  3. Check password strength")
        print("  0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            try:
                length = int(input("  Length [16]: ").strip() or "16")
            except ValueError:
                length = 16
            symbols = input("  Include symbols? [Y/n]: ").strip().lower() != "n"
            upper   = input("  Include uppercase? [Y/n]: ").strip().lower() != "n"
            digits  = input("  Include digits? [Y/n]: ").strip().lower() != "n"
            similar = input("  Exclude similar chars (il1Lo0O)? [y/N]: ").strip().lower() == "y"
            count   = int(input("  How many? [5]: ").strip() or "5")

            print()
            for _ in range(count):
                pwd = generate(length, upper, digits, symbols, similar)
                label, score = strength(pwd)
                bar = "█" * score + "░" * (7 - score)
                print(f"  {pwd}   [{bar}] {label}")

        elif choice == "2":
            words = int(input("  Words [4]: ").strip() or "4")
            sep   = input("  Separator [-]: ").strip() or "-"
            count = int(input("  How many? [3]: ").strip() or "3")
            print()
            for _ in range(count):
                pp = generate_passphrase(words, sep)
                print(f"  {pp}")

        elif choice == "3":
            pwd = input("  Enter password to check: ")
            label, score = strength(pwd)
            bar = "█" * score + "░" * (7 - score)
            print(f"\n  Strength: [{bar}] {label}  ({score}/7)")
            checks = [
                (len(pwd) >= 12,    "Length ≥ 12"),
                (any(c in LOWER  for c in pwd), "Lowercase letters"),
                (any(c in UPPER  for c in pwd), "Uppercase letters"),
                (any(c in DIGITS for c in pwd), "Digits"),
                (any(c in SYMBOLS for c in pwd), "Special characters"),
            ]
            for ok, desc in checks:
                print(f"    {'✓' if ok else '✗'} {desc}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
