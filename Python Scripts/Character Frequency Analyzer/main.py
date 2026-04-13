"""Character Frequency Analyzer — CLI tool.

Analyzes character frequencies in text or a file.
Shows counts, percentages, bar chart, and letter/digit distribution.

Usage:
    python main.py
    python main.py <file.txt>
"""

import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def analyze_chars(text: str, case_sensitive: bool = False) -> dict:
    if not case_sensitive:
        text = text.lower()

    total = len(text)
    counter = Counter(text)

    letters = {k: v for k, v in counter.items() if k.isalpha()}
    digits = {k: v for k, v in counter.items() if k.isdigit()}
    spaces = counter.get(" ", 0) + counter.get("\n", 0) + counter.get("\t", 0)
    punctuation = {k: v for k, v in counter.items() if not k.isalnum() and k not in " \n\t"}

    return {
        "total":       total,
        "counter":     counter,
        "letters":     letters,
        "digits":      digits,
        "spaces":      spaces,
        "punctuation": punctuation,
        "unique":      len(counter),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def bar(count: int, max_count: int, width: int = 30) -> str:
    if max_count == 0:
        return ""
    filled = int((count / max_count) * width)
    return "█" * filled + "░" * (width - filled)


def display_frequency(stats: dict, top_n: int = 30, show_all: bool = False) -> None:
    counter = stats["counter"]
    total = stats["total"]
    sorted_chars = sorted(counter.items(), key=lambda x: -x[1])
    show = sorted_chars if show_all else sorted_chars[:top_n]
    max_count = sorted_chars[0][1] if sorted_chars else 1

    print(f"\n  Total characters: {total:,}  |  Unique: {stats['unique']:,}")
    print(f"  Letters: {sum(stats['letters'].values()):,}  |  "
          f"Digits: {sum(stats['digits'].values()):,}  |  "
          f"Spaces/newlines: {stats['spaces']:,}  |  "
          f"Punctuation: {sum(stats['punctuation'].values()):,}")
    print(f"\n  {'Char':<6} {'Count':>7} {'Pct':>6}  Bar")
    print(f"  {'─' * 55}")

    for char, count in show:
        pct = count / total * 100 if total else 0
        display_char = repr(char) if char in "\n\t\r" else f"'{char}'"
        print(f"  {display_char:<6} {count:>7,}  {pct:>5.1f}%  {bar(count, max_count, 25)}")

    if not show_all and len(sorted_chars) > top_n:
        print(f"  ... {len(sorted_chars) - top_n} more characters")

    # Letter frequency heatmap
    if stats["letters"]:
        print("\n  Letter frequency (A-Z):")
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        max_l = max(stats["letters"].values(), default=1)
        row = ""
        for ch in alphabet:
            cnt = stats["letters"].get(ch, 0)
            shade = "░" if cnt == 0 else "▒" if cnt < max_l * 0.3 else "▓" if cnt < max_l * 0.7 else "█"
            row += f"{ch.upper()}:{shade} "
        print(f"  {row}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_text_from_file(file_path: str) -> str:
    path = Path(file_path.strip('"'))
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    # File argument
    if len(sys.argv) > 1:
        try:
            text = get_text_from_file(sys.argv[1])
            stats = analyze_chars(text, case_sensitive=False)
            display_frequency(stats)
        except FileNotFoundError as e:
            print(e)
        return

    print("Character Frequency Analyzer")
    print("=" * 40)

    while True:
        print("\n1. Analyze typed text")
        print("2. Analyze a file")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            print("  Enter text (type '###' on a new line to finish):")
            lines = []
            while True:
                line = input()
                if line == "###":
                    break
                lines.append(line)
            text = "\n".join(lines)
            if not text.strip():
                print("  No text entered.")
                continue
            case = input("  Case sensitive? (y/n, default n): ").strip().lower() == "y"
            stats = analyze_chars(text, case_sensitive=case)
            display_frequency(stats)

        elif choice == "2":
            file_path = input("  File path: ").strip()
            try:
                text = get_text_from_file(file_path)
                case = input("  Case sensitive? (y/n, default n): ").strip().lower() == "y"
                stats = analyze_chars(text, case_sensitive=case)
                display_frequency(stats)
            except FileNotFoundError as e:
                print(f"  {e}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
