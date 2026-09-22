"""Select one non-empty line from a UTF-8 text file."""

import argparse
import secrets
from pathlib import Path


DEFAULT_LIST = Path(__file__).with_name("file.txt")


def choose_word(path: Path) -> str:
    """Return one non-empty line from the selected word list."""
    words = list(filter(str.strip, path.read_text(encoding="utf-8").splitlines()))
    if not words:
        raise ValueError("word list has no non-empty lines")
    return secrets.choice(words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a random word from a text file.")
    parser.add_argument("word_list", nargs="?", type=Path, default=DEFAULT_LIST)
    args = parser.parse_args()
    if not args.word_list.is_file():
        parser.error("word_list must be an existing text file")
    try:
        print(choose_word(args.word_list))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        parser.exit(1, f"Unable to choose a word: {error}\n")


if __name__ == "__main__":
    main()
