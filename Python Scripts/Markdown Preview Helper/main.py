"""Markdown Preview Helper — CLI tool.

Renders Markdown to terminal-friendly text with basic formatting:
headers, bold, italic, code blocks, lists, blockquotes, links, and
horizontal rules.  Optionally converts to plain text (strips all syntax).

Usage:
    python main.py
    python main.py README.md
"""

import re
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
ITALIC = "\033[3m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
MAGENTA= "\033[35m"
BG_CODE= "\033[100m"  # dark bg for inline code


def _strip_inline(text: str) -> str:
    """Remove inline Markdown syntax (for plain-text mode)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__",     r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"_(.+?)_",       r"\1", text)
    text = re.sub(r"`(.+?)`",       r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text


def _render_inline(text: str) -> str:
    """Apply inline ANSI formatting."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", text)
    text = re.sub(r"__(.+?)__",     lambda m: f"{BOLD}{m.group(1)}{RESET}", text)
    # Italic
    text = re.sub(r"\*(.+?)\*",     lambda m: f"{ITALIC}{m.group(1)}{RESET}", text)
    text = re.sub(r"_([^_]+)_",     lambda m: f"{ITALIC}{m.group(1)}{RESET}", text)
    # Inline code
    text = re.sub(r"`(.+?)`",       lambda m: f"{BG_CODE} {m.group(1)} {RESET}", text)
    # Links
    text = re.sub(r"\[(.+?)\]\((.+?)\)",
                  lambda m: f"{CYAN}{m.group(1)}{RESET} ({DIM}{m.group(2)}{RESET})",
                  text)
    return text


def render_markdown(source: str, plain: bool = False, width: int = 78) -> str:
    """Render Markdown source to a formatted string."""
    lines  = source.splitlines()
    output = []
    i      = 0

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.strip().startswith("```"):
            lang = line.strip().lstrip("`").strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if plain:
                output.extend(code_lines)
            else:
                header = f" code ({lang}) " if lang else " code "
                output.append(f"{DIM}{header.center(width, '─')}{RESET}")
                for cl in code_lines:
                    output.append(f"  {DIM}{cl}{RESET}")
                output.append(f"{DIM}{'─' * width}{RESET}")
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", line):
            output.append(("─" if plain else f"{DIM}{'─' * width}{RESET}"))
            i += 1
            continue

        # ATX headers
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            text  = m.group(2)
            if plain:
                output.append(_strip_inline(text).upper() if level <= 2 else _strip_inline(text))
            else:
                colors = [BOLD + YELLOW, BOLD + YELLOW, BOLD + GREEN,
                          BOLD + MAGENTA, BOLD, BOLD]
                prefix = "#" * level + " "
                output.append(f"{colors[level-1]}{prefix}{_render_inline(text)}{RESET}")
            i += 1
            continue

        # Blockquote
        if line.startswith(">"):
            text = line.lstrip("> ").strip()
            if plain:
                output.append(f"  | {_strip_inline(text)}")
            else:
                output.append(f"  {DIM}│{RESET} {_render_inline(text)}")
            i += 1
            continue

        # Unordered list
        m = re.match(r"^(\s*)[*\-+]\s+(.*)", line)
        if m:
            indent = len(m.group(1))
            text   = m.group(2)
            bullet = "•" if indent == 0 else "◦"
            pad    = "  " * (indent // 2)
            if plain:
                output.append(f"{pad}{bullet} {_strip_inline(text)}")
            else:
                output.append(f"{pad}{CYAN}{bullet}{RESET} {_render_inline(text)}")
            i += 1
            continue

        # Ordered list
        m = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if m:
            indent = len(m.group(1))
            text   = m.group(2)
            num_m  = re.match(r"^(\d+)\.", line.lstrip())
            num    = num_m.group(1) if num_m else "1"
            pad    = "  " * (indent // 2)
            if plain:
                output.append(f"{pad}{num}. {_strip_inline(text)}")
            else:
                output.append(f"{pad}{CYAN}{num}.{RESET} {_render_inline(text)}")
            i += 1
            continue

        # Blank line
        if not line.strip():
            output.append("")
            i += 1
            continue

        # Paragraph
        if plain:
            output.append(textwrap.fill(_strip_inline(line), width=width))
        else:
            output.append(_render_inline(line))
        i += 1

    return "\n".join(output)


def to_plain_text(source: str) -> str:
    return render_markdown(source, plain=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Markdown Preview Helper
-----------------------
1. Preview Markdown (enter text)
2. Preview Markdown file
3. Convert to plain text (enter text)
4. Convert Markdown file to plain text
0. Quit
"""


def read_multiline(label: str) -> str:
    print(f"  Enter {label} (type '###' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "###":
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    # Direct file argument
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            source = p.read_text(encoding="utf-8")
            print(render_markdown(source))
            return
        print(f"File not found: {sys.argv[1]}")
        sys.exit(1)

    print("Markdown Preview Helper")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            source = read_multiline("Markdown text")
            print()
            print(render_markdown(source))

        elif choice == "2":
            path_str = input("  Markdown file path: ").strip().strip('"')
            p = Path(path_str)
            if not p.exists():
                print(f"  File not found: {path_str}")
                continue
            source = p.read_text(encoding="utf-8")
            print()
            print(render_markdown(source))

        elif choice == "3":
            source = read_multiline("Markdown text")
            print()
            print(to_plain_text(source))

        elif choice == "4":
            path_str = input("  Markdown file path: ").strip().strip('"')
            p = Path(path_str)
            if not p.exists():
                print(f"  File not found: {path_str}")
                continue
            source = p.read_text(encoding="utf-8")
            out_path = p.with_suffix(".txt")
            out_path.write_text(to_plain_text(source), encoding="utf-8")
            print(f"\n  Plain text saved to: {out_path}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
