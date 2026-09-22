"""Generate a password with letters, numbers, and special characters."""

import argparse
import secrets
import string


SPECIAL = "@#$%&*"


def generate_password(length: int) -> str:
    """Return a shuffled 50/30/20-style password."""
    letters = length // 2
    numbers = round(length * 0.3)
    symbols = length - letters - numbers
    characters = list(map(lambda _: secrets.choice(string.ascii_letters), range(letters)))
    characters += list(map(lambda _: secrets.choice(string.digits), range(numbers)))
    characters += list(map(lambda _: secrets.choice(SPECIAL), range(symbols)))
    secrets.SystemRandom().shuffle(characters)
    return "".join(characters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mixed-character password.")
    parser.add_argument("length", type=int, help="Password length")
    args = parser.parse_args()
    if args.length < 3:
        parser.error("length must be at least 3")
    print(generate_password(args.length))


if __name__ == "__main__":
    main()
