"""Generate a secure password without whitespace characters."""

import argparse
import secrets
import string


CHARACTERS = string.ascii_letters + string.digits + string.punctuation


def generate_password(length: int) -> str:
    """Return a cryptographically secure random password."""
    return "".join(map(lambda _: secrets.choice(CHARACTERS), range(length)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a secure random password.")
    parser.add_argument("length", type=int, help="Password length")
    args = parser.parse_args()
    if args.length < 1:
        parser.error("length must be at least 1")
    print(generate_password(args.length))


if __name__ == "__main__":
    main()
