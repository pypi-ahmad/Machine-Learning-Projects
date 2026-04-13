"""Text Diff Viewer — CLI tool.

Compare two texts or two files and display a side-by-side or
unified diff with colour-coded additions and deletions.

Usage:
    python main.py
"""

import difflib
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def diff_unified(a: str, b: str, label_a: str = "A", label_b: str = "B",
                 context: int = 3) -> list[str]:
    """Return unified diff lines."""
    lines_a = a.splitlines(keepends=True)
    lines_b = b.splitlines(keepends=True)
    return list(difflib.unified_diff(
        lines_a, lines_b,
        fromfile=label_a, tofile=label_b,
        n=context,
    ))


def diff_side_by_side(a: str, b: str, width: int = 40) -> list[tuple[str, str, str]]:
    """Return list of (left, marker, right) tuples for side-by-side view."""
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines())
    rows: list[tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        left_lines  = a.splitlines()[i1:i2]
        right_lines = b.splitlines()[j1:j2]
        max_len = max(len(left_lines), len(right_lines))
        for k in range(max_len):
            left  = left_lines[k]  if k < len(left_lines)  else ""
            right = right_lines[k] if k < len(right_lines) else ""
            if tag == "equal":
                marker = " "
            elif tag == "replace":
                marker = "|"
            elif tag == "insert":
                marker = ">"
            else:  # delete
                marker = "<"
            rows.append((left, marker, right))
    return rows


def similarity_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_unified(lines: list[str]) -> None:
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("+") and not line.startswith("+++"):
            print(f"  \033[32m{line}\033[0m")
        elif line.startswith("-") and not line.startswith("---"):
            print(f"  \033[31m{line}\033[0m")
        elif line.startswith("@@"):
            print(f"  \033[36m{line}\033[0m")
        else:
            print(f"  {line}")


def print_side_by_side(rows: list[tuple[str, str, str]], width: int = 38) -> None:
    sep = "-" * (width * 2 + 5)
    print(f"  {'LEFT':<{width}}  {'':1}  {'RIGHT'}")
    print(f"  {sep}")
    COLORS = {"|": "\033[33m", ">": "\033[32m", "<": "\033[31m"}
    RESET = "\033[0m"
    for left, marker, right in rows:
        color = COLORS.get(marker, "")
        l = left[:width].ljust(width)
        r = right[:width]
        print(f"  {color}{l}{RESET}  {marker}  {color}{r}{RESET}")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def read_multiline(label: str) -> str:
    print(f"  Enter {label} (type '###' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "###":
            break
        lines.append(line)
    return "\n".join(lines)


def read_file(prompt: str) -> tuple[str, str]:
    path_str = input(prompt).strip().strip('"')
    p = Path(path_str)
    if not p.exists():
        return "", f"[File not found: {path_str}]"
    return p.read_text(encoding="utf-8", errors="replace"), p.name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Text Diff Viewer
----------------
1. Diff two text blocks (unified)
2. Diff two text blocks (side-by-side)
3. Diff two files (unified)
4. Diff two files (side-by-side)
5. Similarity ratio only
0. Quit
"""


def main() -> None:
    print("Text Diff Viewer")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice in ("1", "2"):
            text_a = read_multiline("Text A")
            text_b = read_multiline("Text B")
            ratio  = similarity_ratio(text_a, text_b)
            print(f"\n  Similarity: {ratio:.1%}")
            if choice == "1":
                lines = diff_unified(text_a, text_b, "Text A", "Text B")
                if lines:
                    print_unified(lines)
                else:
                    print("  Texts are identical.")
            else:
                rows = diff_side_by_side(text_a, text_b)
                print_side_by_side(rows)

        elif choice in ("3", "4"):
            text_a, name_a = read_file("  File A path: ")
            text_b, name_b = read_file("  File B path: ")
            if not text_a or not text_b:
                print(f"  Error: {name_a or name_b}")
                continue
            ratio = similarity_ratio(text_a, text_b)
            print(f"\n  Similarity: {ratio:.1%}")
            if choice == "3":
                lines = diff_unified(text_a, text_b, name_a, name_b)
                if lines:
                    print_unified(lines)
                else:
                    print("  Files are identical.")
            else:
                rows = diff_side_by_side(text_a, text_b)
                print_side_by_side(rows)

        elif choice == "5":
            text_a = read_multiline("Text A")
            text_b = read_multiline("Text B")
            ratio  = similarity_ratio(text_a, text_b)
            print(f"\n  Similarity ratio: {ratio:.4f}  ({ratio:.1%})")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
