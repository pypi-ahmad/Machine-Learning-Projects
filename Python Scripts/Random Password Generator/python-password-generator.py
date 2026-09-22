"""Generate one 16-character password."""

import secrets
import string


CHARACTERS = string.ascii_letters + string.digits + string.punctuation


def main() -> None:
    print("".join(map(lambda _: secrets.choice(CHARACTERS), range(16))))


if __name__ == "__main__":
    main()
