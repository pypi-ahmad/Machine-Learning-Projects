"""Regex Tester — CLI tool.

Test regular expressions against text in real time.
Shows matches, groups, named groups, and substitution results.

Usage:
    python main.py
"""

import re
import sys


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def test_regex(pattern: str, text: str, flags: int = 0) -> dict:
    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return {"error": str(e)}

    matches = list(compiled.finditer(text))
    all_matches = []
    for m in matches:
        all_matches.append({
            "match":      m.group(0),
            "start":      m.start(),
            "end":        m.end(),
            "groups":     m.groups(),
            "groupdict":  m.groupdict(),
        })

    return {
        "pattern":    pattern,
        "text":       text,
        "match_count": len(matches),
        "matches":    all_matches,
        "error":      None,
    }


def test_sub(pattern: str, replacement: str, text: str, flags: int = 0) -> str:
    try:
        return re.sub(pattern, replacement, text, flags=flags)
    except re.error as e:
        return f"[Error: {e}]"


def parse_flags(flag_str: str) -> int:
    """Parse flag characters: i=IGNORECASE, m=MULTILINE, s=DOTALL, x=VERBOSE."""
    flags = 0
    for ch in flag_str.lower():
        if ch == "i":
            flags |= re.IGNORECASE
        elif ch == "m":
            flags |= re.MULTILINE
        elif ch == "s":
            flags |= re.DOTALL
        elif ch == "x":
            flags |= re.VERBOSE
    return flags


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def highlight_matches(text: str, matches: list[dict]) -> str:
    """Return text with [MATCH] markers around matched spans."""
    if not matches:
        return text
    result = []
    last = 0
    for m in matches:
        result.append(text[last:m["start"]])
        result.append(f"[{m['match']}]")
        last = m["end"]
    result.append(text[last:])
    return "".join(result)


def display_result(result: dict) -> None:
    if result.get("error"):
        print(f"\n  ✗ Regex error: {result['error']}")
        return

    count = result["match_count"]
    text = result["text"]
    matches = result["matches"]

    print(f"\n  Matches found: {count}")

    if count == 0:
        print("  No matches.")
        return

    # Highlighted
    print(f"  Highlighted: {highlight_matches(text, matches)}")
    print()

    for i, m in enumerate(matches[:20], 1):
        print(f"  Match {i}: '{m['match']}'  span=({m['start']}, {m['end']})")
        if m["groups"]:
            print(f"           Groups: {m['groups']}")
        if m["groupdict"]:
            print(f"           Named:  {m['groupdict']}")

    if count > 20:
        print(f"  ... and {count - 20} more matches")


# ---------------------------------------------------------------------------
# Cheat sheet
# ---------------------------------------------------------------------------

CHEAT_SHEET = """
Regex Quick Reference
─────────────────────
.        Any character (except newline by default)
^        Start of string (or line with MULTILINE)
$        End of string (or line with MULTILINE)
*        0 or more
+        1 or more
?        0 or 1
{n}      Exactly n times
{n,m}    Between n and m times
[abc]    Character class
[^abc]   Negated class
\\d       Digit [0-9]
\\D       Non-digit
\\w       Word char [a-zA-Z0-9_]
\\W       Non-word
\\s       Whitespace
\\S       Non-whitespace
\\b       Word boundary
(abc)    Capturing group
(?:abc)  Non-capturing group
(?P<name>abc)  Named group
|        Alternation (a|b)
(?=...)  Positive lookahead
(?!...)  Negative lookahead

Flags: i=ignore case, m=multiline, s=dot all, x=verbose
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Regex Tester
------------
1. Test pattern against text
2. Substitution (re.sub)
3. Show cheat sheet
0. Quit
"""


def get_multiline_text() -> str:
    print("  Enter text (type '###' on new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "###":
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    print("Regex Tester  (Python re module)")

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            pattern = input("  Regex pattern: ").strip()
            if not pattern:
                continue
            flag_str = input("  Flags (i/m/s/x, blank=none): ").strip()
            flags = parse_flags(flag_str)
            text = get_multiline_text()
            result = test_regex(pattern, text, flags)
            display_result(result)

        elif choice == "2":
            pattern = input("  Regex pattern: ").strip()
            if not pattern:
                continue
            replacement = input("  Replacement string: ")
            flag_str = input("  Flags (i/m/s/x, blank=none): ").strip()
            flags = parse_flags(flag_str)
            text = get_multiline_text()
            result = test_sub(pattern, replacement, text, flags)
            print(f"\n  Result:\n  {result}")

        elif choice == "3":
            print(CHEAT_SHEET)

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
