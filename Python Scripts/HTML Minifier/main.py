"""HTML Minifier — CLI developer tool.

Minify HTML files by removing whitespace, comments, and
optionally inlining CSS/JS. Shows size reduction stats.

Usage:
    python main.py index.html
    python main.py index.html --output dist/index.html
    python main.py index.html --no-comments --aggressive
"""

import argparse
import os
import re
import sys

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Minification pipeline ──────────────────────────────────────────────────────

def remove_html_comments(html: str, keep_ie: bool = True) -> str:
    """Remove HTML comments, optionally keeping IE conditional comments."""
    if keep_ie:
        # Keep <!--[if IE]> ... <![endif]--> blocks
        return re.sub(r"<!--(?!\[if).*?-->", "", html, flags=re.DOTALL)
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def minify_inline_css(html: str) -> str:
    """Minify CSS within <style> tags."""
    def process_style(m):
        css = m.group(1)
        # Remove CSS comments
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
        # Remove whitespace around ; : { }
        css = re.sub(r"\s*([;:{}])\s*", r"\1", css)
        css = re.sub(r"\s+", " ", css)
        css = css.strip()
        return f"<style>{css}</style>"
    return re.sub(r"<style[^>]*>(.*?)</style>", process_style, html,
                  flags=re.DOTALL | re.IGNORECASE)


def minify_inline_js(html: str) -> str:
    """Basic minification of JS within <script> tags."""
    def process_script(m):
        js = m.group(1)
        # Remove // line comments (not inside strings — best effort)
        js = re.sub(r"//[^\n]*", "", js)
        # Remove /* block comments */
        js = re.sub(r"/\*.*?\*/", "", js, flags=re.DOTALL)
        # Collapse whitespace
        js = re.sub(r"\s+", " ", js)
        js = re.sub(r"\s*([;,{}()=+\-*/<>!&|])\s*", r"\1", js)
        js = js.strip()
        return f"<script>{js}</script>"
    return re.sub(r"<script[^>]*>(.*?)</script>", process_script, html,
                  flags=re.DOTALL | re.IGNORECASE)


def collapse_whitespace(html: str, aggressive: bool = False) -> str:
    """Collapse consecutive whitespace between tags."""
    # Remove newlines and extra spaces between tags
    html = re.sub(r">\s+<", "><", html)
    if aggressive:
        # Also collapse spaces inside text nodes (dangerous for <pre>)
        html = re.sub(r"\s{2,}", " ", html)
    else:
        html = re.sub(r" {2,}", " ", html)
        html = re.sub(r"\n{2,}", "\n", html)
    return html.strip()


def remove_optional_attributes(html: str) -> str:
    """Remove optional quotes around simple attribute values (aggressive)."""
    # type="text/javascript" → remove entirely
    html = re.sub(r'\s+type=["\']text/javascript["\']', "", html, flags=re.IGNORECASE)
    html = re.sub(r'\s+type=["\']text/css["\']',        "", html, flags=re.IGNORECASE)
    # Remove empty attributes like ' class=""'
    html = re.sub(r'\s+\w+=""',                          "",  html)
    return html


def minify_html(html: str, remove_comments: bool = True,
                minify_css: bool = True, minify_js: bool = True,
                aggressive: bool = False, keep_ie_comments: bool = True) -> str:
    if remove_comments:
        html = remove_html_comments(html, keep_ie=keep_ie_comments)
    if minify_css:
        html = minify_inline_css(html)
    if minify_js:
        html = minify_inline_js(html)
    if aggressive:
        html = remove_optional_attributes(html)
    html = collapse_whitespace(html, aggressive=aggressive)
    return html


def human_size(n: int) -> str:
    if n < 1024:       return f"{n} B"
    if n < 1024**2:    return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"


def print_stats(original: str, minified: str):
    orig_sz = len(original.encode("utf-8"))
    mini_sz = len(minified.encode("utf-8"))
    saved   = orig_sz - mini_sz
    ratio   = saved / orig_sz * 100 if orig_sz else 0
    print(c(f"\n  Original : {human_size(orig_sz)}", "dim"))
    print(c(f"  Minified : {human_size(mini_sz)}", "dim"))
    print(c(f"  Saved    : {human_size(saved)} ({ratio:.1f}%)", "green"))


def interactive_mode():
    print(c("HTML Minifier\n", "bold"))
    print("Commands: file <path>, paste, quit")
    print("Options: --no-comments, --no-css, --no-js, --aggressive\n")

    while True:
        try:
            line = input(c("html> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower() if parts else ""
        flags = set(parts[1:])

        def get_html_content(cmd, parts):
            if cmd == "file" and len(parts) > 1:
                try:
                    with open(parts[1], encoding="utf-8") as f:
                        return f.read(), parts[1]
                except FileNotFoundError:
                    print(c(f"  File not found: {parts[1]}", "red"))
                    return None, None
            elif cmd == "paste":
                print("Paste HTML (type END on a new line to finish):")
                lines = []
                while True:
                    try:
                        l = input()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if l.strip() == "END":
                        break
                    lines.append(l)
                return "\n".join(lines), None
            return None, None

        html, fpath = get_html_content(cmd, parts)
        if html is None:
            if cmd not in ("file", "paste", "quit", "exit", "q"):
                print(c("  Unknown command.", "yellow"))
            continue

        aggressive = "--aggressive" in flags
        minified   = minify_html(
            html,
            remove_comments="--no-comments" not in flags,
            minify_css="--no-css" not in flags,
            minify_js="--no-js" not in flags,
            aggressive=aggressive,
        )
        print_stats(html, minified)

        save = input(c("  Save to file? [y/N]: ", "cyan")).strip().lower()
        if save == "y":
            out = input(c("  Output path: ", "cyan")).strip()
            if out:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(minified)
                print(c(f"  ✓ Saved to {out}", "green"))
        else:
            print(c("\n─── Minified Output ─────────────────────", "dim"))
            print(minified[:500] + ("..." if len(minified) > 500 else ""))


def main():
    parser = argparse.ArgumentParser(description="Minify HTML files")
    parser.add_argument("input",          nargs="?",      help="Input HTML file")
    parser.add_argument("--output", "-o", metavar="FILE", help="Output file")
    parser.add_argument("--no-comments",  action="store_true", dest="no_comments")
    parser.add_argument("--no-css",       action="store_true", dest="no_css")
    parser.add_argument("--no-js",        action="store_true", dest="no_js")
    parser.add_argument("--aggressive",   action="store_true")
    parser.add_argument("--stats",        action="store_true", help="Show stats only")
    args = parser.parse_args()

    if args.input:
        try:
            with open(args.input, encoding="utf-8") as f:
                html = f.read()
        except FileNotFoundError:
            print(c(f"File not found: {args.input}", "red"))
            sys.exit(1)

        minified = minify_html(
            html,
            remove_comments=not args.no_comments,
            minify_css=not args.no_css,
            minify_js=not args.no_js,
            aggressive=args.aggressive,
        )
        print_stats(html, minified)

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
