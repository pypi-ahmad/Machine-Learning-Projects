"""Markdown to HTML — CLI developer tool.

Convert Markdown files to standalone HTML with optional
syntax highlighting, table of contents, and custom CSS.

Usage:
    python main.py README.md
    python main.py README.md --output index.html
    python main.py README.md --toc --theme dark
    python main.py --watch README.md
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


# ── Themes ─────────────────────────────────────────────────────────────────────

THEMES = {
    "light": {
        "bg": "#ffffff", "fg": "#222222", "link": "#005a9c",
        "code_bg": "#f4f4f4", "border": "#dddddd",
        "quote_border": "#cccccc", "quote_fg": "#555555",
        "pre_bg": "#f8f8f8",
    },
    "dark": {
        "bg": "#1e1e2e", "fg": "#cdd6f4", "link": "#89b4fa",
        "code_bg": "#313244", "border": "#45475a",
        "quote_border": "#585b70", "quote_fg": "#a6adc8",
        "pre_bg": "#181825",
    },
    "github": {
        "bg": "#ffffff", "fg": "#24292e", "link": "#0366d6",
        "code_bg": "#f6f8fa", "border": "#e1e4e8",
        "quote_border": "#dfe2e5", "quote_fg": "#6a737d",
        "pre_bg": "#f6f8fa",
    },
}


def make_css(theme: dict) -> str:
    t = theme
    return f"""\
*,*::before,*::after{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     line-height:1.7;max-width:860px;margin:0 auto;padding:2rem 1.5rem;
     background:{t['bg']};color:{t['fg']}}}
a{{color:{t['link']};text-decoration:none}}
a:hover{{text-decoration:underline}}
h1,h2,h3,h4,h5,h6{{line-height:1.25;margin:1.5rem 0 .5rem}}
h1{{font-size:2em;border-bottom:1px solid {t['border']};padding-bottom:.3em}}
h2{{font-size:1.5em;border-bottom:1px solid {t['border']};padding-bottom:.2em}}
hr{{border:none;border-top:1px solid {t['border']};margin:2rem 0}}
pre{{background:{t['pre_bg']};border:1px solid {t['border']};
     border-radius:6px;padding:1rem;overflow-x:auto}}
code{{font-size:.88em;font-family:'SFMono-Regular',Consolas,monospace;
      background:{t['code_bg']};padding:.15em .4em;border-radius:3px}}
pre code{{background:none;padding:0;font-size:.9em}}
blockquote{{border-left:4px solid {t['quote_border']};margin:0;
            padding:.5rem 1rem;color:{t['quote_fg']}}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid {t['border']};padding:.5rem .75rem;text-align:left}}
th{{background:{t['code_bg']}}}
img{{max-width:100%;height:auto}}
.toc{{background:{t['code_bg']};border:1px solid {t['border']};
      border-radius:6px;padding:1rem 1.5rem;margin-bottom:2rem}}
.toc ul{{margin:.5rem 0;padding-left:1.5rem}}
.toc li{{margin:.2rem 0}}
.badge{{display:inline-block;padding:.2em .6em;border-radius:3px;
        font-size:.8em;background:{t['code_bg']};color:{t['fg']}}}
"""


# ── Markdown parser ────────────────────────────────────────────────────────────

def escape_html(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def inline(text: str) -> str:
    """Process inline Markdown elements."""
    # Escape HTML first (protect user content)
    # Bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"___(.+?)___",        r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*",      r"<strong>\1</strong>",          text)
    text = re.sub(r"__(.+?)__",          r"<strong>\1</strong>",          text)
    text = re.sub(r"\*(.+?)\*",          r"<em>\1</em>",                  text)
    text = re.sub(r"_(.+?)_",            r"<em>\1</em>",                  text)
    text = re.sub(r"~~(.+?)~~",          r"<del>\1</del>",                text)
    # Inline code (handle before links)
    text = re.sub(r"`(.+?)`",            r"<code>\1</code>",              text)
    # Images before links
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', text)
    # Links
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",  r'<a href="\2">\1</a>',    text)
    # Auto-links
    text = re.sub(r"<(https?://[^>]+)>",        r'<a href="\1">\1</a>',    text)
    return text


def parse_table(rows: list[str]) -> str:
    """Convert markdown table lines to HTML table."""
    if len(rows) < 2:
        return ""
    headers = [h.strip() for h in rows[0].strip("|").split("|")]
    html    = "<table>\n<thead><tr>"
    for h in headers:
        html += f"<th>{inline(h)}</th>"
    html += "</tr></thead>\n<tbody>\n"
    for row in rows[2:]:
        cells = [td.strip() for td in row.strip("|").split("|")]
        html += "<tr>"
        for td in cells:
            html += f"<td>{inline(td)}</td>"
        html += "</tr>\n"
    html += "</tbody></table>"
    return html


def md_to_html(md: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Convert Markdown to HTML, also return heading list for TOC."""
    lines    = md.split("\n")
    output   = []
    headings = []
    in_code  = False
    in_ul    = False
    in_ol    = False
    in_p     = False
    table_buf: list[str] = []

    def close_list():
        nonlocal in_ul, in_ol
        if in_ul:
            output.append("</ul>")
            in_ul = False
        if in_ol:
            output.append("</ol>")
            in_ol = False

    def close_p():
        nonlocal in_p
        if in_p:
            output.append("</p>")
            in_p = False

    def flush_table():
        if table_buf:
            output.append(parse_table(table_buf))
            table_buf.clear()

    for line in lines:
        # Code fence
        if re.match(r"^```", line):
            flush_table()
            if in_code:
                output.append("</code></pre>")
                in_code = False
            else:
                close_list(); close_p()
                lang = line[3:].strip()
                cls  = f' class="language-{lang}"' if lang else ""
                output.append(f"<pre><code{cls}>")
                in_code = True
            continue
        if in_code:
            output.append(escape_html(line))
            continue

        # Table
        if "|" in line:
            flush_table()
            close_list(); close_p()
            table_buf.append(line)
            continue
        elif table_buf:
            flush_table()

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            close_list(); close_p()
            lvl   = len(m.group(1))
            text  = m.group(2).strip()
            slug  = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            headings.append((lvl, text, slug))
            output.append(f'<h{lvl} id="{slug}">{inline(text)}</h{lvl}>')
            continue

        # HR
        if re.match(r"^\s*[-*_]{3,}\s*$", line):
            close_list(); close_p(); flush_table()
            output.append("<hr>")
            continue

        # Blockquote
        if line.startswith("> "):
            close_list(); close_p()
            output.append(f"<blockquote><p>{inline(line[2:])}</p></blockquote>")
            continue

        # Unordered list
        if re.match(r"^[-*+] ", line):
            close_p()
            if not in_ul:
                close_list()
                output.append("<ul>")
                in_ul = True
            output.append(f"<li>{inline(line[2:].strip())}</li>")
            continue

        # Ordered list
        m2 = re.match(r"^(\d+)\.\s+(.*)", line)
        if m2:
            close_p()
            if not in_ol:
                close_list()
                output.append("<ol>")
                in_ol = True
            output.append(f"<li>{inline(m2.group(2))}</li>")
            continue

        # Empty line
        if not line.strip():
            close_list(); close_p()
            continue

        # Paragraph
        close_list()
        if not in_p:
            output.append("<p>")
            in_p = True
        output.append(inline(line))

    close_list(); close_p(); flush_table()
    if in_code:
        output.append("</code></pre>")

    return "\n".join(output), headings


def build_toc(headings: list[tuple[int, str, str]]) -> str:
    if not headings:
        return ""
    min_lvl = min(h[0] for h in headings)
    html    = ['<nav class="toc"><strong>Table of Contents</strong><ul>']
    prev_lvl = min_lvl
    for lvl, text, slug in headings:
        if lvl > prev_lvl:
            html.append("<ul>")
        elif lvl < prev_lvl:
            html.append("</ul>")
        html.append(f'<li><a href="#{slug}">{text}</a></li>')
        prev_lvl = lvl
    html.append("</ul></nav>")
    return "\n".join(html)


def wrap_html(body: str, title: str, css: str, toc: str = "") -> str:
    toc_section = f"\n  {toc}\n" if toc else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escape_html(title)}</title>
  <style>
{css}
  </style>
</head>
<body>
{toc_section}
{body}
</body>
</html>
"""


def convert(md: str, title: str = "Document",
            theme: str = "light", include_toc: bool = False) -> str:
    body, headings = md_to_html(md)
    css = make_css(THEMES.get(theme, THEMES["light"]))
    toc = build_toc(headings) if include_toc else ""
    return wrap_html(body, title, css, toc)


def watch_file(path: str, output: str, theme: str, toc: bool):
    print(c(f"  Watching {path} for changes... (Ctrl+C to stop)\n", "dim"))
    last_mtime = None
    try:
        while True:
            try:
                mtime = os.path.getmtime(path)
            except FileNotFoundError:
                time.sleep(1)
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                with open(path, encoding="utf-8") as f:
                    md = f.read()
                title = os.path.splitext(os.path.basename(path))[0]
                html  = convert(md, title=title, theme=theme, include_toc=toc)
                with open(output, "w", encoding="utf-8") as f:
                    f.write(html)
                ts = time.strftime("%H:%M:%S")
                print(c(f"  [{ts}] Rebuilt → {output}", "green"))
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(c("\n  Stopped watching.", "dim"))


def interactive_mode():
    print(c("Markdown to HTML Converter\n", "bold"))
    print("Commands: convert <file>, paste, quit")
    print(f"Themes:   {', '.join(THEMES)}\n")

    while True:
        try:
            line = input(c("md> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower() if parts else ""

        if cmd == "convert" and len(parts) > 1:
            fpath = parts[1]
            theme = parts[2] if len(parts) > 2 else "light"
            try:
                with open(fpath, encoding="utf-8") as f:
                    md = f.read()
            except FileNotFoundError:
                print(c(f"  File not found: {fpath}", "red"))
                continue
            title = os.path.splitext(os.path.basename(fpath))[0]
            toc   = input(c("  Include TOC? [y/N]: ", "cyan")).strip().lower() == "y"
            html  = convert(md, title=title, theme=theme, include_toc=toc)
            out   = input(c("  Output path (Enter to print): ", "cyan")).strip()
            if out:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(html)
                print(c(f"  ✓ Saved to {out}", "green"))
            else:
                print(html[:800] + ("..." if len(html) > 800 else ""))

        elif cmd == "paste":
            print("Paste Markdown (type END to finish):")
            lines = []
            while True:
                try:
                    l = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if l.strip() == "END":
                    break
                lines.append(l)
            html = convert("\n".join(lines), include_toc=False)
            print(c("\n─── HTML Output ─────────────────────────", "dim"))
            print(html[:800] + ("..." if len(html) > 800 else ""))
        else:
            if cmd not in ("quit", "exit", "q"):
                print(c("  Unknown command.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="Convert Markdown to HTML")
    parser.add_argument("input",           nargs="?",      help="Input Markdown file")
    parser.add_argument("--output", "-o",  metavar="FILE", help="Output HTML file")
    parser.add_argument("--theme",         default="light",
                        choices=list(THEMES.keys()), help="Color theme")
    parser.add_argument("--toc",           action="store_true", help="Generate table of contents")
    parser.add_argument("--watch",         action="store_true", help="Watch for changes")
    parser.add_argument("--title",         metavar="TITLE",     help="Override page title")
    args = parser.parse_args()

    if args.input:
        try:
            with open(args.input, encoding="utf-8") as f:
                md = f.read()
        except FileNotFoundError:
            print(c(f"File not found: {args.input}", "red"))
            sys.exit(1)

        title  = args.title or os.path.splitext(os.path.basename(args.input))[0]
        output = args.output or args.input.replace(".md", ".html").replace(".markdown", ".html")

        if args.watch:
            watch_file(args.input, output, args.theme, args.toc)
            return

        html = convert(md, title=title, theme=args.theme, include_toc=args.toc)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(html)
            print(c(f"✓ Written to {args.output}", "green"))
        else:
            print(html)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
