"""Crossword Helper — CLI tool.

Find words matching a pattern (with known and unknown letters).
Supports wildcards, anagram solving, and word length filters.

Usage:
    python main.py
    python main.py --pattern "c_t"
    python main.py --anagram "listen"
    python main.py --pattern "p_th_n" --min 5
"""

import argparse
import re
import sys
from pathlib import Path


# Built-in word list (common English words) — used when no system dict found
BUILTIN_WORDS = """
the be to of and a in that have it for not on with he as you do at
this but his by from they we say her she or an will my one all would
there their what so up out if about who get which go me when make can
like time no just him know take people into year your good some could
them see other than then now look only come its over think also back
after use two how our work first well way even new want because any
these give day most us cat dog run man men woman old big small house
great long little own right place world very still nation hand high
every never money away each form without through car turn back give
live look word off ask where never also should try friend left number
set put end does large must big turn right might come show good play
small number off always turn end move live give play spell air away
animal house point page letter mother answer found study still learn
should world above between need large often hand move thing tell does
city play small number off always turn end cat dog fox box top hot
open close fast slow heavy light hard soft dark bright run walk jump
fly swim climb sleep eat drink cook read write speak sing laugh cry
think dream hope love hate fear fight rest work play dance sing clap
python java code class loop while break return import print function
blue red green yellow orange purple white black gray pink brown
one two three four five six seven eight nine ten hundred thousand
zero plus minus times divide equal less more than above below
start stop begin end left right up down north south east west
time year day week month morning night winter summer spring fall
fire water earth wind space star moon sun cloud rain snow ice
key door room house city country world ocean sea river mountain
word book page story poem song music note sound voice
"""


def load_words(min_len: int = 2) -> set[str]:
    """Load word list from system dictionary or built-in fallback."""
    for path in ["/usr/share/dict/words", "/usr/dict/words"]:
        p = Path(path)
        if p.exists():
            words = set(w.strip().lower() for w in p.read_text().splitlines()
                        if w.strip().isalpha() and len(w.strip()) >= min_len)
            return words

    # Fallback: built-in list
    words = set(w.strip().lower() for w in BUILTIN_WORDS.split()
                if w.strip().isalpha() and len(w.strip()) >= min_len)
    return words


def pattern_match(pattern: str, words: set[str]) -> list[str]:
    """
    Match words against a pattern.
    Use _ or ? for unknown letter, * for any number of letters.
    Example: c_t → cat, cut, cot
    """
    pat = pattern.lower()
    # Convert to regex: _ or ? → single char, * → any chars
    rx = pat.replace(".", r"\.").replace("_", "[a-z]").replace("?", "[a-z]").replace("*", "[a-z]*")
    rx = f"^{rx}$"
    try:
        compiled = re.compile(rx)
    except re.error as e:
        raise ValueError(f"Invalid pattern: {e}")
    return sorted(w for w in words if compiled.match(w))


def anagram_solve(letters: str, words: set[str],
                  min_len: int = 3, exact: bool = False) -> list[str]:
    """Find all words that are anagrams (or sub-anagrams) of the given letters."""
    letters = letters.lower()
    from collections import Counter
    avail = Counter(letters)
    results = []
    for w in words:
        if exact and len(w) != len(letters): continue
        if len(w) < min_len: continue
        wc = Counter(w)
        if all(wc[c] <= avail[c] for c in wc):
            results.append(w)
    return sorted(results, key=lambda x: (-len(x), x))


def interactive(words: set[str]) -> None:
    print("=== Crossword Helper ===")
    print("Commands: pattern | anagram | contains | length | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break

        elif cmd == "pattern":
            p = input("  Pattern (use _ for unknown letter, * for any): ").strip()
            results = pattern_match(p, words)
            print(f"  {len(results)} match(es): {', '.join(results[:50])}")
            if len(results) > 50: print(f"  … and {len(results)-50} more")

        elif cmd == "anagram":
            letters = input("  Letters: ").strip()
            exact   = input("  Exact length only? (y/n): ").strip().lower() == "y"
            results = anagram_solve(letters, words, exact=exact)
            print(f"  {len(results)} anagram(s): {', '.join(results[:30])}")

        elif cmd == "contains":
            sub = input("  Must contain substring: ").strip().lower()
            length = input("  Word length (or blank for any): ").strip()
            results = sorted(w for w in words if sub in w
                             and (not length or len(w) == int(length)))
            print(f"  {len(results)} match(es): {', '.join(results[:50])}")
            if len(results) > 50: print(f"  … and {len(results)-50} more")

        elif cmd == "length":
            ln = int(input("  Exact word length: ").strip())
            results = sorted(w for w in words if len(w) == ln)
            print(f"  {len(results)} words of length {ln}: {', '.join(results[:50])}")

        else:
            print("  Commands: pattern | anagram | contains | length | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Crossword Helper")
    parser.add_argument("--pattern",  metavar="P",   help="Pattern with _ wildcards")
    parser.add_argument("--anagram",  metavar="STR", help="Find anagrams of these letters")
    parser.add_argument("--contains", metavar="SUB", help="Words containing substring")
    parser.add_argument("--length",   type=int,      help="Filter by word length")
    parser.add_argument("--min",      type=int, default=2, help="Minimum word length")
    parser.add_argument("--exact",    action="store_true",
                        help="Exact anagram only (same letter count)")
    args = parser.parse_args()

    print("  Loading word list…", end="", flush=True)
    words = load_words(args.min)
    print(f" {len(words):,} words loaded.")

    if args.pattern:
        results = pattern_match(args.pattern, words)
        if args.length: results = [w for w in results if len(w) == args.length]
        print(f"{len(results)} match(es): {', '.join(results[:80])}")
    elif args.anagram:
        results = anagram_solve(args.anagram, words, args.min, args.exact)
        if args.length: results = [w for w in results if len(w) == args.length]
        print(f"{len(results)} anagram(s): {', '.join(results[:60])}")
    elif args.contains:
        results = sorted(w for w in words if args.contains.lower() in w
                         and (not args.length or len(w) == args.length))
        print(f"{len(results)} match(es): {', '.join(results[:80])}")
    else:
        interactive(words)


if __name__ == "__main__":
    main()
