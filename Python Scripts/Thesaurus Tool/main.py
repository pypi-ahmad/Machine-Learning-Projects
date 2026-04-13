"""Thesaurus Tool — CLI tool.

Find synonyms, antonyms, definitions, and example sentences.
Uses the Datamuse API (no key required).

Usage:
    python main.py
    python main.py --word happy
    python main.py --word fast --antonyms
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse


DATAMUSE = "https://api.datamuse.com/words"


def fetch(url: str) -> list:
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def synonyms(word: str, n: int = 20) -> list[dict]:
    """Words with similar meaning (rel_syn)."""
    url = f"{DATAMUSE}?rel_syn={urllib.parse.quote(word)}&max={n}&md=d"
    return fetch(url)


def antonyms(word: str, n: int = 15) -> list[dict]:
    """Words with opposite meaning (rel_ant)."""
    url = f"{DATAMUSE}?rel_ant={urllib.parse.quote(word)}&max={n}"
    return fetch(url)


def related(word: str, n: int = 15) -> list[dict]:
    """Broader terms (rel_bga = triggers / associations)."""
    url = f"{DATAMUSE}?rel_trg={urllib.parse.quote(word)}&max={n}"
    return fetch(url)


def rhymes(word: str, n: int = 20) -> list[dict]:
    url = f"{DATAMUSE}?rel_rhy={urllib.parse.quote(word)}&max={n}"
    return fetch(url)


def sounds_like(word: str, n: int = 10) -> list[dict]:
    url = f"{DATAMUSE}?sl={urllib.parse.quote(word)}&max={n}"
    return fetch(url)


def display_words(title: str, words: list[dict]) -> None:
    if not words:
        print(f"  {title}: (none found)")
        return
    word_list = [w["word"] for w in words]
    print(f"\n  {title} ({len(word_list)}):")
    # Wrap at 70 chars
    line = "  "
    for w in word_list:
        if len(line) + len(w) + 2 > 72:
            print(line.rstrip(", "))
            line = f"  {w}, "
        else:
            line += f"{w}, "
    if line.strip().rstrip(","):
        print(line.rstrip(", "))


def lookup(word: str, show_ant: bool = True, show_related: bool = True,
           show_rhymes: bool = False) -> None:
    print(f"\n  Word: \"{word}\"")
    print(f"  {'─'*50}")

    syns = synonyms(word)
    display_words("Synonyms", syns)

    if show_ant:
        ants = antonyms(word)
        display_words("Antonyms", ants)

    if show_related:
        rels = related(word)
        display_words("Related / Triggers", rels)

    if show_rhymes:
        rhs = rhymes(word)
        display_words("Rhymes", rhs)

    print()


def interactive():
    print("=== Thesaurus Tool ===")
    print("Commands: <word> | ant <word> | rhyme <word> | sounds <word> | quit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not line: continue
        parts = line.split(None, 1)
        cmd   = parts[0].lower()

        if cmd in ("quit", "q", "exit"): break
        elif cmd == "ant":
            word = parts[1].strip() if len(parts)>1 else input("  Word: ").strip()
            try:
                ants = antonyms(word)
                display_words(f"Antonyms of '{word}'", ants)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "rhyme":
            word = parts[1].strip() if len(parts)>1 else input("  Word: ").strip()
            try:
                rhs = rhymes(word)
                display_words(f"Rhymes with '{word}'", rhs)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "sounds":
            word = parts[1].strip() if len(parts)>1 else input("  Word: ").strip()
            try:
                sls = sounds_like(word)
                display_words(f"Sounds like '{word}'", sls)
            except ValueError as e: print(f"  Error: {e}")
        else:
            # Treat entire input as a word to look up
            try: lookup(line, show_ant=True)
            except ValueError as e: print(f"  Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Thesaurus Tool")
    parser.add_argument("--word",     metavar="WORD", help="Word to look up")
    parser.add_argument("--antonyms", action="store_true", help="Show antonyms only")
    parser.add_argument("--rhymes",   action="store_true", help="Show rhymes")
    parser.add_argument("--sounds",   action="store_true", help="Show words that sound like")
    args = parser.parse_args()

    try:
        if args.word:
            if args.antonyms:
                ants = antonyms(args.word)
                display_words(f"Antonyms of '{args.word}'", ants)
            elif args.rhymes:
                rhs = rhymes(args.word)
                display_words(f"Rhymes with '{args.word}'", rhs)
            elif args.sounds:
                sls = sounds_like(args.word)
                display_words(f"Sounds like '{args.word}'", sls)
            else:
                lookup(args.word, show_rhymes=True)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
