"""Print case-insensitive words that occur exactly once in a text file.

Usage:
    uv run python unique.py
    uv run python unique.py path\\to\\document.txt
"""

import argparse
import re
from pathlib import Path


def unique_words(path: Path) -> list[str]:
    """Return sorted alphabetic words that appear exactly once in *path*."""
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            for word in re.findall(r"[\w]+", line.lower()):
                counts[word] = counts.get(word, 0) + 1

    words = []
    for word, count in counts.items():
        if count == 1:
            words.append(word)
    return sorted(words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print words that occur exactly once.")
    parser.add_argument("file", nargs="?", help="UTF-8 text file to inspect")
    args = parser.parse_args()
    path = Path(args.file) if args.file else Path(__file__).with_name("text_file.txt")
    if not path.is_file():
        parser.error(f"file not found: {path}")
    print(unique_words(path))


if __name__ == "__main__":
    main()
