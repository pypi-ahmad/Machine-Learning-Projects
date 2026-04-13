"""Slug Generator — CLI tool.

Converts text to URL-friendly slugs.
Supports multiple slug styles, Unicode transliteration, custom separators,
max-length truncation, and batch generation.

Usage:
    python main.py
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

_TRANSLITERATION = {
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
    "æ": "ae", "ç": "c", "è": "e", "é": "e", "ê": "e", "ë": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i", "ð": "d", "ñ": "n",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o",
    "ù": "u", "ú": "u", "û": "u", "ü": "u", "ý": "y", "þ": "th",
    "ß": "ss", "ÿ": "y", "&": "and", "@": "at", "%": "percent",
    "+": "plus", "#": "hash", "$": "dollar",
}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "as",
}


def _transliterate(text: str) -> str:
    """Replace known special characters and strip remaining diacritics."""
    result = []
    for ch in text.lower():
        if ch in _TRANSLITERATION:
            result.append(_TRANSLITERATION[ch])
        else:
            result.append(ch)
    # Normalize remaining accented chars
    normalized = unicodedata.normalize("NFD", "".join(result))
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def generate_slug(
    text: str,
    separator: str = "-",
    max_length: int = 0,
    remove_stop_words: bool = False,
    lowercase: bool = True,
) -> str:
    """Generate a URL slug from text."""
    s = _transliterate(text)
    # Keep only alphanumeric, spaces, separator
    s = re.sub(r"[^\w\s]", " ", s)
    words = s.split()
    if remove_stop_words:
        words = [w for w in words if w.lower() not in STOP_WORDS] or words
    if lowercase:
        words = [w.lower() for w in words]
    slug = separator.join(words)
    # Collapse repeated separators
    sep_escaped = re.escape(separator)
    slug = re.sub(f"{sep_escaped}{{2,}}", separator, slug)
    slug = slug.strip(separator)
    if max_length and len(slug) > max_length:
        slug = slug[:max_length].rstrip(separator)
    return slug


def generate_variants(text: str) -> dict[str, str]:
    """Generate common slug variants."""
    return {
        "kebab-case    ": generate_slug(text, "-"),
        "snake_case    ": generate_slug(text, "_"),
        "dot.case      ": generate_slug(text, "."),
        "no-stop-words ": generate_slug(text, "-", remove_stop_words=True),
        "max-50 chars  ": generate_slug(text, "-", max_length=50),
        "UPPER-KEBAB   ": generate_slug(text, "-", lowercase=False).upper(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Slug Generator
--------------
1. Generate slug (custom options)
2. Show all variants
3. Batch generate (multiple titles)
0. Quit
"""


def main() -> None:
    print("Slug Generator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            text = input("  Text: ").strip()
            if not text:
                continue
            sep = input("  Separator (default '-'): ").strip() or "-"
            max_len_str = input("  Max length (0 = unlimited): ").strip()
            max_len = int(max_len_str) if max_len_str.isdigit() else 0
            rm_stop = input("  Remove stop words? (y/n, default n): ").strip().lower() == "y"
            slug = generate_slug(text, sep, max_len, rm_stop)
            print(f"\n  Slug: {slug}")
            print(f"  Length: {len(slug)}")

        elif choice == "2":
            text = input("  Text: ").strip()
            if not text:
                continue
            print()
            for label, slug in generate_variants(text).items():
                print(f"  {label}: {slug}")

        elif choice == "3":
            print("  Enter titles (one per line, blank line to finish):")
            titles = []
            while True:
                t = input("  > ").strip()
                if not t:
                    break
                titles.append(t)
            if not titles:
                continue
            sep = input("  Separator (default '-'): ").strip() or "-"
            print()
            print(f"  {'Original':<35}  Slug")
            print(f"  {'-'*35}  {'-'*40}")
            for t in titles:
                slug = generate_slug(t, sep)
                print(f"  {t[:35]:<35}  {slug}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
