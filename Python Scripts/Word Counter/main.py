"""Word Counter — CLI tool.

Counts words, characters, lines, sentences, and paragraphs
in typed text or a file. Shows readability estimates.

Usage:
    python main.py
    python main.py <file.txt>
"""

import sys
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    lines = text.splitlines()
    non_empty_lines = [l for l in lines if l.strip()]
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p for p in paragraphs if p.strip()]
    chars_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

    # Average word length
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0

    # Flesch reading ease estimate
    syllables = sum(_count_syllables(w) for w in words)
    if len(sentences) > 0 and len(words) > 0:
        flesch = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
        flesch = max(0, min(100, flesch))
    else:
        flesch = None

    # Most common words (top 10)
    from collections import Counter
    stopwords = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","it","this","that","was","are","be"}
    word_counts = Counter(w.lower() for w in words if w.lower() not in stopwords and len(w) > 2)
    top_words = word_counts.most_common(10)

    return {
        "characters":       len(text),
        "chars_no_spaces":  chars_no_spaces,
        "words":            len(words),
        "lines":            len(lines),
        "non_empty_lines":  len(non_empty_lines),
        "sentences":        len(sentences),
        "paragraphs":       len(paragraphs),
        "avg_word_length":  avg_word_len,
        "syllables":        syllables,
        "flesch_score":     flesch,
        "top_words":        top_words,
        "unique_words":     len(set(w.lower() for w in words)),
    }


def _count_syllables(word: str) -> int:
    """Rough syllable count for English words."""
    word = word.lower().rstrip("e")
    vowels = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowels))


def reading_time(words: int) -> str:
    """Estimate reading time at 200 words per minute."""
    minutes = words / 200
    if minutes < 1:
        secs = int(minutes * 60)
        return f"~{secs} seconds"
    return f"~{minutes:.1f} minutes"


def flesch_label(score: float) -> str:
    if score >= 90: return "Very easy"
    if score >= 80: return "Easy"
    if score >= 70: return "Fairly easy"
    if score >= 60: return "Standard"
    if score >= 50: return "Fairly difficult"
    if score >= 30: return "Difficult"
    return "Very confusing"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_stats(stats: dict) -> None:
    print(f"\n  {'─' * 40}")
    print(f"  Characters      : {stats['characters']:,}")
    print(f"  Chars (no space): {stats['chars_no_spaces']:,}")
    print(f"  Words           : {stats['words']:,}  (unique: {stats['unique_words']:,})")
    print(f"  Lines           : {stats['lines']:,}  (non-empty: {stats['non_empty_lines']:,})")
    print(f"  Sentences       : {stats['sentences']:,}")
    print(f"  Paragraphs      : {stats['paragraphs']:,}")
    print(f"  Avg word length : {stats['avg_word_length']:.1f} chars")
    print(f"  Syllables       : {stats['syllables']:,}")
    print(f"  Reading time    : {reading_time(stats['words'])}")
    if stats["flesch_score"] is not None:
        print(f"  Readability     : {stats['flesch_score']:.1f} ({flesch_label(stats['flesch_score'])})")
    if stats["top_words"]:
        top = ", ".join(f"{w}({n})" for w, n in stats["top_words"][:5])
        print(f"  Top words       : {top}")
    print(f"  {'─' * 40}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # File argument
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"Analyzing: {path.name}")
        display_stats(analyze_text(text))
        return

    print("Word Counter")
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
            if text.strip():
                display_stats(analyze_text(text))
            else:
                print("  No text entered.")

        elif choice == "2":
            file_path = input("  File path: ").strip().strip('"')
            path = Path(file_path)
            if not path.exists():
                print(f"  File not found: {path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"  Analyzing: {path.name}")
            display_stats(analyze_text(text))

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
