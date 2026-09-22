"""Generate placeholder email-like addresses for examples and test data."""

import argparse
import secrets
import string


DOMAINS = ("gmail", "yahoo", "comcast", "verizon", "charter", "hotmail", "outlook", "frontier")
EXTENSIONS = ("com", "net", "org", "gov")
CHARACTERS = string.ascii_lowercase + string.digits


def make_email() -> str:
    """Create one random email-like address."""
    username = "".join(map(lambda _: secrets.choice(CHARACTERS), range(secrets.randbelow(20) + 1)))
    return f"{username}@{secrets.choice(DOMAINS)}.{secrets.choice(EXTENSIONS)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate placeholder email-like addresses.")
    parser.add_argument("count", type=int, help="Number of addresses to generate")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be at least 1")
    print("\n".join(map(lambda _: make_email(), range(args.count))))


if __name__ == "__main__":
    main()
