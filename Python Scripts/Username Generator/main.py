"""Username Generator — CLI tool.

Generates creative, unique usernames from real names, keywords,
or random combinations.  Supports multiple styles and availability
filtering (length, allowed characters).

Usage:
    python main.py
"""

import random
import re
import string


# ---------------------------------------------------------------------------
# Word banks
# ---------------------------------------------------------------------------

ADJECTIVES = [
    "swift", "bold", "calm", "dark", "epic", "fair", "grim", "hard",
    "iron", "jade", "keen", "lone", "mild", "neon", "pale", "rash",
    "sage", "tall", "vast", "wild", "zero", "azure", "brave", "crisp",
    "dusk", "elite", "frost", "ghost", "hyper", "icy", "jetblack",
    "lunar", "mystic", "noble", "onyx", "pixel", "quick", "robo",
    "solar", "turbo", "ultra", "vapor", "wired", "xenon", "yellow",
]

NOUNS = [
    "ace", "arc", "axe", "bat", "byte", "code", "dash", "echo", "flux",
    "gear", "hawk", "iris", "jet", "king", "lark", "maze", "node",
    "orb", "pike", "quiz", "rex", "sage", "tide", "unit", "vox",
    "wave", "xray", "yak", "zone", "arrow", "blade", "cloud", "drift",
    "ember", "forge", "glyph", "helix", "index", "joker", "karma",
    "lemma", "matrix", "nexus", "orbit", "pulse", "quark", "raven",
    "sigma", "token", "umbra", "vista", "warden", "xenon", "yield",
]

VERBS = [
    "blink", "burst", "catch", "craft", "crush", "debug", "drift",
    "forge", "glide", "grind", "launch", "morph", "parse", "patch",
    "probe", "query", "roast", "route", "slice", "solve", "spark",
    "stack", "surge", "trace", "twist",
]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def from_name(first: str, last: str, style: str = "auto") -> list[str]:
    f, l = _clean(first), _clean(last)
    results = []
    if not f and not l:
        return results

    num = str(random.randint(10, 99))
    year = str(random.randint(90, 99))

    if f and l:
        results += [
            f"{f}.{l}",
            f"{f}_{l}",
            f"{f[0]}{l}",
            f"{f}{l[0].upper()}",
            f"{f}{l}{num}",
            f"{f[0]}{l}{year}",
            f"{l}{f[0]}",
            f"{f[:3]}{l[:3]}",
        ]
    elif f:
        results += [f"{f}{num}", f"{f}_{num}", f"the{f}"]
    else:
        results += [f"{l}{num}", f"{l}_{num}"]
    return list(dict.fromkeys(results))  # unique, order-preserving


def from_keyword(keyword: str) -> list[str]:
    kw  = _clean(keyword)
    num = str(random.randint(1, 999))
    adj = random.choice(ADJECTIVES)
    noun= random.choice(NOUNS)
    return [
        f"{kw}_{num}",
        f"{adj}_{kw}",
        f"{kw}_{noun}",
        f"{noun}_{kw}",
        f"{kw}{num}",
        f"the{kw}",
        f"{adj}{kw.capitalize()}",
        f"{kw}x{num}",
    ]


def random_usernames(count: int = 10, style: str = "mixed") -> list[str]:
    results = []
    for _ in range(count * 3):  # generate extras to deduplicate
        num  = str(random.randint(1, 9999))
        adj  = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        verb = random.choice(VERBS)
        r    = random.choice([
            f"{adj}{noun}",
            f"{adj}_{noun}",
            f"{verb}{noun}",
            f"{noun}{num}",
            f"{adj}{num}",
            f"{verb}_{noun}_{num}",
            f"{adj}{noun}{num[-2:]}",
        ])
        results.append(r)
    seen = set()
    unique = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique.append(r)
        if len(unique) >= count:
            break
    return unique


def filter_usernames(
    usernames: list[str],
    min_len: int = 3,
    max_len: int = 20,
    allow_underscore: bool = True,
    allow_dot: bool = False,
    allow_hyphen: bool = False,
    allow_numbers: bool = True,
) -> list[str]:
    pattern_chars = "a-z0-9"
    if allow_underscore:
        pattern_chars += "_"
    if allow_dot:
        pattern_chars += r"\."
    if allow_hyphen:
        pattern_chars += r"\-"
    if not allow_numbers:
        pattern_chars = pattern_chars.replace("0-9", "")
    pat = re.compile(f"^[{pattern_chars}]{{{{min_len},{max_len}}}}\$")
    pat = re.compile(f"^[{pattern_chars}]{{{min_len},{max_len}}}$")
    return [u for u in usernames if pat.match(u)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Username Generator
------------------
1. Generate from name
2. Generate from keyword
3. Random username ideas
4. All three + filter
0. Quit
"""


def main() -> None:
    print("Username Generator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            first = input("  First name: ").strip()
            last  = input("  Last name : ").strip()
            names = from_name(first, last)
            if names:
                print(f"\n  Suggestions ({len(names)}):")
                for n in names:
                    print(f"    {n}")
            else:
                print("  Enter at least one name.")

        elif choice == "2":
            kw = input("  Keyword: ").strip()
            if not kw:
                continue
            names = from_keyword(kw)
            print(f"\n  Suggestions ({len(names)}):")
            for n in names:
                print(f"    {n}")

        elif choice == "3":
            count_str = input("  How many? (default 10): ").strip()
            count = int(count_str) if count_str.isdigit() else 10
            names = random_usernames(count)
            print(f"\n  {count} random usernames:")
            for n in names:
                print(f"    {n}")

        elif choice == "4":
            first = input("  First name (optional): ").strip()
            last  = input("  Last name  (optional): ").strip()
            kw    = input("  Keyword    (optional): ").strip()
            all_names = []
            if first or last:
                all_names += from_name(first, last)
            if kw:
                all_names += from_keyword(kw)
            all_names += random_usernames(10)

            print("\n  Filter options (press Enter for defaults):")
            min_l = int(input("  Min length (default 3) : ").strip() or "3")
            max_l = int(input("  Max length (default 20): ").strip() or "20")
            allow_num = input("  Allow numbers? (y/n, default y): ").strip().lower() != "n"
            allow_us  = input("  Allow underscore? (y/n, default y): ").strip().lower() != "n"

            filtered = filter_usernames(all_names, min_l, max_l,
                                        allow_underscore=allow_us,
                                        allow_numbers=allow_num)
            if filtered:
                print(f"\n  {len(filtered)} matching username(s):")
                for n in filtered:
                    print(f"    {n}")
            else:
                print("\n  No names passed the filter. Try relaxing the constraints.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
