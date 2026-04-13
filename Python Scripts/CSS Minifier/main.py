"""CSS Minifier — CLI developer tool.

Minify CSS files by removing comments, collapsing whitespace,
and optionally applying shorthand optimizations.

Usage:
    python main.py style.css
    python main.py style.css --output dist/style.css
    python main.py style.css --aggressive
    python main.py --watch style.css
"""

import argparse
import os
import re
import sys
import time

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Minification pipeline ──────────────────────────────────────────────────────

def remove_comments(css: str) -> str:
    """Remove /* ... */ CSS comments."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def collapse_whitespace(css: str) -> str:
    """Collapse all whitespace sequences to single spaces."""
    css = re.sub(r"\s+", " ", css)
    return css.strip()


def remove_whitespace_around_symbols(css: str) -> str:
    """Remove spaces around CSS structural symbols."""
    css = re.sub(r"\s*([{};:,>~+])\s*", r"\1", css)
    # Restore space after colon in selector pseudo-classes (not properties)
    # This is a best-effort approach
    return css


def remove_trailing_semicolons(css: str) -> str:
    """Remove the last semicolon before closing brace."""
    return re.sub(r";\s*}", "}", css)


def normalize_colors(css: str) -> str:
    """Shorten 6-digit hex colors to 3-digit where possible."""
    def shorten_hex(m):
        h = m.group(1)
        if h[0] == h[1] and h[2] == h[3] and h[4] == h[5]:
            return f"#{h[0]}{h[2]}{h[4]}"
        return m.group(0)
    return re.sub(r"#([0-9a-fA-F]{6})\b", shorten_hex, css)


def normalize_zeros(css: str) -> str:
    """Remove units from zero values and leading zeros from decimals."""
    # 0px, 0em, 0rem, 0% → 0
    css = re.sub(r"\b0(px|em|rem|%|pt|pc|cm|mm|in|ex|ch|vw|vh|vmin|vmax)\b", "0", css)
    # 0.5 → .5
    css = re.sub(r"(?<![0-9])0\.(\d)", r".\1", css)
    return css


def remove_empty_rules(css: str) -> str:
    """Remove rules with empty declaration blocks."""
    return re.sub(r"[^{}]+\{\s*\}", "", css)


def aggressive_optimize(css: str) -> str:
    """Additional aggressive optimizations."""
    # Remove spaces inside parentheses
    css = re.sub(r"\(\s+", "(", css)
    css = re.sub(r"\s+\)", ")", css)
    # Remove space after comma in functions like rgba()
    css = re.sub(r",\s+", ",", css)
    return css


def minify_css(css: str, aggressive: bool = False,
               shorten_colors: bool = True, shorten_zeros: bool = True) -> str:
    css = remove_comments(css)
    css = collapse_whitespace(css)
    css = remove_whitespace_around_symbols(css)
    css = remove_trailing_semicolons(css)
    if shorten_colors:
        css = normalize_colors(css)
    if shorten_zeros:
        css = normalize_zeros(css)
    css = remove_empty_rules(css)
    if aggressive:
        css = aggressive_optimize(css)
    return css.strip()


def human_size(n: int) -> str:
    if n < 1024:     return f"{n} B"
    if n < 1024**2:  return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"


def print_stats(original: str, minified: str):
    orig_sz = len(original.encode("utf-8"))
    mini_sz = len(minified.encode("utf-8"))
    saved   = orig_sz - mini_sz
    ratio   = saved / orig_sz * 100 if orig_sz else 0
    print(c(f"\n  Original : {human_size(orig_sz)}", "dim"))
    print(c(f"  Minified : {human_size(mini_sz)}", "dim"))
    print(c(f"  Saved    : {human_size(saved)} ({ratio:.1f}%)", "green"))


def lint_css(css: str) -> list[str]:
    """Basic CSS linting checks."""
    issues = []
    # Count braces
    opens  = css.count("{")
    closes = css.count("}")
    if opens != closes:
        issues.append(f"⚠ Unbalanced braces: {opens} {{ vs {closes} }}")
    # Detect !important overuse
    count = len(re.findall(r"!important", css, re.IGNORECASE))
    if count > 3:
        issues.append(f"⚠ Excessive !important usage ({count} occurrences).")
    # Detect vendor prefixes without standard property
    if re.search(r"-webkit-|-moz-|-ms-|-o-", css):
        issues.append("ℹ Vendor prefixes detected — consider autoprefixer.")
    # Detect very long selectors
    for sel in re.findall(r"([^{}]+)\{", css):
        if len(sel.strip()) > 200:
            issues.append("⚠ Very long selector detected (may indicate specificity issues).")
            break
    if not issues:
        issues.append("✓ No obvious issues found.")
    return issues


def watch_file(path: str, output: str = None, aggressive: bool = False):
    """Watch a CSS file and re-minify on changes."""
    print(c(f"  Watching {path} for changes... (Ctrl+C to stop)\n", "dim"))
    last_mtime = None
    try:
        while True:
            try:
                mtime = os.path.getmtime(path)
            except FileNotFoundError:
                print(c(f"  File not found: {path}", "red"))
                time.sleep(1)
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                with open(path, encoding="utf-8") as f:
                    css = f.read()
                minified = minify_css(css, aggressive=aggressive)
                orig_sz = len(css.encode("utf-8"))
                mini_sz = len(minified.encode("utf-8"))
                ratio   = (orig_sz - mini_sz) / orig_sz * 100 if orig_sz else 0
                ts = time.strftime("%H:%M:%S")
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        f.write(minified)
                    print(c(f"  [{ts}] Rebuilt → {output}  "
                            f"({human_size(orig_sz)} → {human_size(mini_sz)}, -{ratio:.1f}%)", "green"))
                else:
                    print(c(f"  [{ts}] Changed  "
                            f"({human_size(orig_sz)} → {human_size(mini_sz)}, -{ratio:.1f}%)", "cyan"))
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(c("\n  Stopped watching.", "dim"))


def interactive_mode():
    print(c("CSS Minifier\n", "bold"))
    print("Commands: file <path>, paste, lint <path>, quit")
    print("Options:  --aggressive, --no-colors, --no-zeros\n")

    while True:
        try:
            line = input(c("css> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower() if parts else ""
        flags = set(parts[1:]) if len(parts) > 1 else set()

        if cmd == "file" and len(parts) > 1:
            fpath = parts[1]
            flags = set(parts[2:])
            try:
                with open(fpath, encoding="utf-8") as f:
                    css = f.read()
            except FileNotFoundError:
                print(c(f"  File not found: {fpath}", "red"))
                continue
        elif cmd == "paste":
            print("Paste CSS (type END on a new line to finish):")
            lines = []
            while True:
                try:
                    l = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if l.strip() == "END":
                    break
                lines.append(l)
            css   = "\n".join(lines)
            fpath = None
        elif cmd == "lint" and len(parts) > 1:
            try:
                with open(parts[1], encoding="utf-8") as f:
                    css = f.read()
                for issue in lint_css(css):
                    col = "red" if "⚠" in issue else ("cyan" if "ℹ" in issue else "green")
                    print(c(f"  {issue}", col))
            except FileNotFoundError:
                print(c(f"  File not found: {parts[1]}", "red"))
            continue
        else:
            if cmd not in ("quit", "exit", "q"):
                print(c("  Unknown command.", "yellow"))
            continue

        aggressive   = "--aggressive" in flags
        shorten_col  = "--no-colors" not in flags
        shorten_zero = "--no-zeros" not in flags

        minified = minify_css(css, aggressive=aggressive,
                              shorten_colors=shorten_col, shorten_zeros=shorten_zero)
        print_stats(css, minified)

        save = input(c("  Save to file? [y/N]: ", "cyan")).strip().lower()
        if save == "y":
            out = input(c("  Output path: ", "cyan")).strip()
            if out:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(minified)
                print(c(f"  ✓ Saved to {out}", "green"))
        else:
            print(c("\n─── Minified Output ─────────────────────", "dim"))
            print(minified[:600] + ("..." if len(minified) > 600 else ""))


def main():
    parser = argparse.ArgumentParser(description="Minify CSS files")
    parser.add_argument("input",           nargs="?",      help="Input CSS file")
    parser.add_argument("--output", "-o",  metavar="FILE", help="Output file")
    parser.add_argument("--aggressive",    action="store_true")
    parser.add_argument("--no-colors",     action="store_true", dest="no_colors")
    parser.add_argument("--no-zeros",      action="store_true", dest="no_zeros")
    parser.add_argument("--lint",          action="store_true", help="Lint only")
    parser.add_argument("--stats",         action="store_true", help="Show stats only")
    parser.add_argument("--watch",         action="store_true", help="Watch for changes")
    args = parser.parse_args()

    if args.input:
        try:
            with open(args.input, encoding="utf-8") as f:
                css = f.read()
        except FileNotFoundError:
            print(c(f"File not found: {args.input}", "red"))
            sys.exit(1)

        if args.lint:
            for issue in lint_css(css):
                col = "red" if "⚠" in issue else ("cyan" if "ℹ" in issue else "green")
                print(c(issue, col))
            return

        if args.watch:
            watch_file(args.input, output=args.output, aggressive=args.aggressive)
            return

        minified = minify_css(css, aggressive=args.aggressive,
                              shorten_colors=not args.no_colors,
                              shorten_zeros=not args.no_zeros)
        print_stats(css, minified)

        if not args.stats:
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(minified)
                print(c(f"✓ Written to {args.output}", "green"))
            else:
                print(minified)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
