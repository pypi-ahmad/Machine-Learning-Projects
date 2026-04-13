"""Static Site Generator — CLI developer tool.

Generate a static HTML website from Markdown files and a
simple template system. Supports front matter, layouts,
and an asset copy step.

Usage:
    python main.py init my-site
    python main.py build
    python main.py build --src content --out public
    python main.py new post "My First Post"
    python main.py serve
"""

import argparse
import http.server
import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "dim": "\033[2m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Markdown → HTML ────────────────────────────────────────────────────────────

def md_to_html(md: str) -> str:
    """Convert a subset of Markdown to HTML."""
    lines   = md.split("\n")
    output  = []
    in_code = False
    in_ul   = False
    in_ol   = False
    in_p    = False

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

    def inline(text):
        # Bold + italic
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
        text = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>",           text)
        text = re.sub(r"\*(.+?)\*",          r"<em>\1</em>",                  text)
        text = re.sub(r"`(.+?)`",            r"<code>\1</code>",              text)
        # Links
        text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>',         text)
        # Images
        text = re.sub(r"!\[(.+?)\]\((.+?)\)", r'<img src="\2" alt="\1">',    text)
        return text

    for line in lines:
        # Code fence
        if line.startswith("```"):
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
            output.append(line)
            continue

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            close_list(); close_p()
            lvl  = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            output.append(f'<h{lvl} id="{slug}">{inline(m.group(2))}</h{lvl}>')
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", line.strip()):
            close_list(); close_p()
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
            output.append(f"<li>{inline(line[2:])}</li>")
            continue

        # Ordered list
        if re.match(r"^\d+\. ", line):
            close_p()
            if not in_ol:
                close_list()
                output.append("<ol>")
                in_ol = True
            output.append(f"<li>{inline(re.sub(r'^\d+\. ', '', line))}</li>")
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

    close_list(); close_p()
    if in_code:
        output.append("</code></pre>")

    return "\n".join(output)


# ── Front matter ───────────────────────────────────────────────────────────────

def parse_front_matter(text: str) -> tuple[dict, str]:
    """Extract YAML-like front matter delimited by --- lines."""
    meta = {}
    if not text.startswith("---"):
        return meta, text
    end = text.find("\n---", 3)
    if end == -1:
        return meta, text
    front = text[3:end].strip()
    body  = text[end+4:].lstrip("\n")
    for line in front.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


# ── Default template ───────────────────────────────────────────────────────────

DEFAULT_LAYOUT = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="stylesheet" href="{root}assets/style.css">
</head>
<body>
  <header>
    <nav><a href="{root}index.html">{site_name}</a></nav>
  </header>
  <main>
    <article>
      <h1>{title}</h1>
      <p class="meta">{date}</p>
      {content}
    </article>
  </main>
  <footer><p>{site_name} — Built with Static Site Generator</p></footer>
</body>
</html>
"""

DEFAULT_INDEX_LAYOUT = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{site_name}</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <header>
    <nav><strong>{site_name}</strong></nav>
  </header>
  <main>
    <h1>Welcome to {site_name}</h1>
    <ul class="post-list">
{post_links}
    </ul>
  </main>
  <footer><p>{site_name} — Built with Static Site Generator</p></footer>
</body>
</html>
"""

DEFAULT_CSS = """\
*,*::before,*::after{box-sizing:border-box}
body{font-family:system-ui,sans-serif;line-height:1.6;max-width:800px;
     margin:0 auto;padding:1rem 1.5rem;color:#222}
header{border-bottom:1px solid #ddd;padding:.5rem 0;margin-bottom:2rem}
nav a{text-decoration:none;font-weight:700;color:#005a9c}
h1,h2,h3{line-height:1.2}
pre{background:#f4f4f4;padding:1rem;overflow-x:auto;border-radius:4px}
code{font-size:.9em;background:#f4f4f4;padding:.1em .3em;border-radius:3px}
pre code{background:none;padding:0}
blockquote{border-left:4px solid #ccc;margin-left:0;padding-left:1rem;color:#555}
.meta{color:#888;font-size:.9em}
ul.post-list{list-style:none;padding:0}
ul.post-list li{margin:.5rem 0}
footer{border-top:1px solid #ddd;margin-top:3rem;padding:.5rem 0;color:#888;font-size:.85em}
"""

CONFIG_FILE = "ssg_config.json"


def load_config(root: str = ".") -> dict:
    path = os.path.join(root, CONFIG_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"site_name": "My Site", "src": "content", "out": "public"}


def save_config(cfg: dict, root: str = "."):
    with open(os.path.join(root, CONFIG_FILE), "w") as f:
        json.dump(cfg, f, indent=2)


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_init(name: str):
    if os.path.exists(name):
        print(c(f"Directory '{name}' already exists.", "yellow"))
    else:
        os.makedirs(name)

    cfg = {"site_name": name.replace("-", " ").title(), "src": "content", "out": "public"}
    save_config(cfg, name)

    content_dir = os.path.join(name, "content")
    assets_dir  = os.path.join(name, "assets")
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(assets_dir,  exist_ok=True)

    # Starter post
    starter = f"""---
title: Hello World
date: {datetime.today().strftime('%Y-%m-%d')}
---

Welcome to **{cfg['site_name']}**!

This is your first post. Edit `content/hello-world.md` to get started.

## What is this?

A static site generated from Markdown files.
"""
    with open(os.path.join(content_dir, "hello-world.md"), "w") as f:
        f.write(starter)

    print(c(f"✓ Initialized site at '{name}/'", "green"))
    print(f"  Edit {name}/content/*.md to write posts.")
    print(f"  Run: python main.py build --src {name}/content --out {name}/public")


def cmd_new(kind: str, title: str):
    cfg   = load_config()
    slug  = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    fname = f"{slug}.md"
    path  = os.path.join(cfg["src"], fname)
    os.makedirs(cfg["src"], exist_ok=True)
    if os.path.exists(path):
        print(c(f"File already exists: {path}", "yellow"))
        return
    content = f"""---
title: {title}
date: {datetime.today().strftime('%Y-%m-%d')}
---

Write your {kind} here.
"""
    with open(path, "w") as f:
        f.write(content)
    print(c(f"✓ Created {path}", "green"))


def cmd_build(src: str, out: str, cfg: dict):
    if not os.path.isdir(src):
        print(c(f"Source directory not found: {src}", "red"))
        sys.exit(1)

    os.makedirs(out, exist_ok=True)
    assets_out = os.path.join(out, "assets")
    os.makedirs(assets_out, exist_ok=True)

    # Write default CSS if none exists
    css_out = os.path.join(assets_out, "style.css")
    if not os.path.exists(css_out):
        with open(css_out, "w") as f:
            f.write(DEFAULT_CSS)

    # Copy assets if src has an assets folder
    src_assets = os.path.join(os.path.dirname(src), "assets")
    if os.path.isdir(src_assets):
        for item in os.listdir(src_assets):
            shutil.copy2(os.path.join(src_assets, item), assets_out)

    pages  = []
    count  = 0

    for fname in sorted(os.listdir(src)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(src, fname)
        with open(fpath, encoding="utf-8") as f:
            raw = f.read()

        meta, body = parse_front_matter(raw)
        title      = meta.get("title", fname.replace(".md", "").replace("-", " ").title())
        date       = meta.get("date", "")
        slug       = fname.replace(".md", "")
        html_body  = md_to_html(body)

        page_html = DEFAULT_LAYOUT.format(
            title=title, date=date,
            content=html_body,
            site_name=cfg.get("site_name", "My Site"),
            root="",
        )

        out_path = os.path.join(out, f"{slug}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page_html)

        pages.append({"slug": slug, "title": title, "date": date})
        count += 1
        print(c(f"  ✓ {fname} → {slug}.html", "dim"))

    # Sort pages by date desc
    pages.sort(key=lambda p: p["date"], reverse=True)
    post_links = "\n".join(
        f'      <li><span class="meta">{p["date"]}</span> '
        f'<a href="{p["slug"]}.html">{p["title"]}</a></li>'
        for p in pages
    )

    index_html = DEFAULT_INDEX_LAYOUT.format(
        site_name=cfg.get("site_name", "My Site"),
        post_links=post_links,
    )
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print(c(f"\n  Built {count} page(s) + index → {out}/", "green"))


def cmd_serve(out: str, port: int = 8000):
    if not os.path.isdir(out):
        print(c(f"Output directory not found: {out}  (run 'build' first)", "yellow"))
        return

    os.chdir(out)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(c(f"  {args[0]} {args[1]}", "dim"))

    server = http.server.HTTPServer(("", port), Handler)
    print(c(f"  Serving {out}/ at http://localhost:{port}", "cyan"))
    print(c("  Press Ctrl+C to stop.\n", "dim"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(c("\n  Server stopped.", "dim"))


def main():
    parser = argparse.ArgumentParser(description="Static site generator")
    sub    = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Initialize a new site")
    p_init.add_argument("name", help="Site directory name")

    p_new = sub.add_parser("new", help="Create a new page/post")
    p_new.add_argument("kind",  choices=["post", "page"])
    p_new.add_argument("title", help="Page title")

    p_build = sub.add_parser("build", help="Build the site")
    p_build.add_argument("--src",  default=None, help="Source Markdown directory")
    p_build.add_argument("--out",  default=None, help="Output HTML directory")

    p_serve = sub.add_parser("serve", help="Serve the built site locally")
    p_serve.add_argument("--out",  default=None)
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args.name)

    elif args.command == "new":
        cmd_new(args.kind, args.title)

    elif args.command == "build":
        cfg = load_config()
        src = args.src or cfg.get("src", "content")
        out = args.out or cfg.get("out", "public")
        cmd_build(src, out, cfg)

    elif args.command == "serve":
        cfg  = load_config()
        out  = args.out or cfg.get("out", "public")
        cmd_serve(out, port=args.port)

    else:
        # Interactive quick-build
        print(c("Static Site Generator\n", "bold"))
        cfg = load_config()
        print(f"  Config: src={cfg.get('src','content')}  out={cfg.get('out','public')}")
        ans = input(c("  Build now? [Y/n]: ", "cyan")).strip().lower()
        if ans in ("", "y"):
            cmd_build(cfg.get("src", "content"), cfg.get("out", "public"), cfg)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
