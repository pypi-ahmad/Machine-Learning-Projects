"""Code Snippet Manager — CLI developer tool.

Store, search, tag, and copy code snippets from the command line.
Snippets are saved to a local JSON file.

Usage:
    python main.py
    python main.py add
    python main.py search <query>
    python main.py list
    python main.py show <id>
    python main.py copy <id>
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snippets.json")

ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "magenta": "\033[95m",
        "reset": "\033[0m", "dim": "\033[2m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


LANGUAGES = ["python", "javascript", "typescript", "bash", "sql", "html",
             "css", "rust", "go", "java", "c", "cpp", "json", "yaml", "other"]


def load() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save(snippets: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(snippets, f, indent=2)


def next_id(snippets: list[dict]) -> int:
    return max((s["id"] for s in snippets), default=0) + 1


def copy_to_clipboard(text: str) -> bool:
    try:
        if sys.platform == "win32":
            import subprocess
            proc = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            return True
        elif sys.platform == "darwin":
            import subprocess
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            return True
        else:
            import subprocess
            for cmd in [["xclip", "-selection", "clipboard"],
                        ["xsel", "--clipboard", "--input"]]:
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                    proc.communicate(text.encode("utf-8"))
                    return True
                except FileNotFoundError:
                    continue
    except Exception:
        pass
    return False


def print_snippet(s: dict, show_code: bool = True):
    print(f"\n  {c(f'#{s[\"id\"]}', 'cyan')} {c(s['title'], 'bold')}  "
          f"{c(s.get('language',''), 'yellow')}  "
          f"{c(' '.join('#'+t for t in s.get('tags',[])), 'magenta')}")
    print(f"  {c(s.get('description',''), 'dim')}")
    print(f"  {c('Created: '+s.get('created',''), 'dim')}")
    if show_code:
        print(c("  ─" * 35, "dim"))
        for line in s["code"].splitlines():
            print(f"  {line}")
        print(c("  ─" * 35, "dim"))


def cmd_add(args):
    snippets = load()
    print(c("\n  Add New Snippet\n", "bold"))

    def ask(prompt, default=""):
        val = input(c(f"  {prompt}" + (f" [{default}]" if default else "") + ": ", "cyan")).strip()
        return val if val else default

    title   = ask("Title")
    if not title:
        print(c("  Title is required.", "red"))
        return
    lang    = ask("Language", "python")
    desc    = ask("Description (optional)", "")
    tags_in = ask("Tags (comma-separated)", "")
    tags    = [t.strip() for t in tags_in.split(",") if t.strip()]

    print(c("  Enter code (type END on a new line to finish):", "cyan"))
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "END":
            break
        lines.append(line)
    code = "\n".join(lines)

    snippet = {
        "id":          next_id(snippets),
        "title":       title,
        "language":    lang,
        "description": desc,
        "tags":        tags,
        "code":        code,
        "created":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    snippets.append(snippet)
    save(snippets)
    print(c(f"\n  ✓ Saved snippet #{snippet['id']}: {title}", "green"))


def cmd_list(args):
    snippets = load()
    if not snippets:
        print(c("  No snippets yet. Use 'add' to create one.", "yellow"))
        return

    lang_filter = getattr(args, "lang", None)
    tag_filter  = getattr(args, "tag", None)

    print(c(f"\n  {len(snippets)} snippet(s)\n", "bold"))
    for s in snippets:
        if lang_filter and s.get("language", "").lower() != lang_filter.lower():
            continue
        if tag_filter and tag_filter not in s.get("tags", []):
            continue
        tags_str = " ".join(c(f"#{t}", "magenta") for t in s.get("tags", []))
        print(f"  {c(f'#{s[\"id\"]:>3}', 'cyan')} {c(s['title'][:40], 'bold'):<42} "
              f"{c(s.get('language',''):>12, 'yellow')}  {tags_str}")


def cmd_show(args):
    snippets = load()
    target   = next((s for s in snippets if s["id"] == args.id), None)
    if not target:
        print(c(f"  Snippet #{args.id} not found.", "red"))
        return
    print_snippet(target, show_code=True)


def cmd_search(args):
    snippets = load()
    q        = args.query.lower()
    results  = []
    for s in snippets:
        if (q in s["title"].lower() or
            q in s.get("description", "").lower() or
            q in s["code"].lower() or
            any(q in t for t in s.get("tags", []))):
            results.append(s)
    if not results:
        print(c(f"  No snippets matching '{args.query}'.", "yellow"))
        return
    print(c(f"\n  Found {len(results)} snippet(s):\n", "bold"))
    for s in results:
        print_snippet(s, show_code=False)


def cmd_copy(args):
    snippets = load()
    target   = next((s for s in snippets if s["id"] == args.id), None)
    if not target:
        print(c(f"  Snippet #{args.id} not found.", "red"))
        return
    ok = copy_to_clipboard(target["code"])
    if ok:
        print(c(f"  ✓ Snippet #{args.id} copied to clipboard.", "green"))
    else:
        print(c("  Could not access clipboard. Here is the code:", "yellow"))
        print(target["code"])


def cmd_delete(args):
    snippets = load()
    target   = next((s for s in snippets if s["id"] == args.id), None)
    if not target:
        print(c(f"  Snippet #{args.id} not found.", "red"))
        return
    confirm = input(c(f"  Delete snippet #{args.id} '{target['title']}'? [y/N]: ", "yellow")).strip().lower()
    if confirm == "y":
        snippets = [s for s in snippets if s["id"] != args.id]
        save(snippets)
        print(c(f"  ✓ Deleted snippet #{args.id}.", "green"))


def cmd_edit(args):
    snippets = load()
    idx = next((i for i, s in enumerate(snippets) if s["id"] == args.id), None)
    if idx is None:
        print(c(f"  Snippet #{args.id} not found.", "red"))
        return
    s = snippets[idx]
    print_snippet(s, show_code=True)
    print(c("\n  Press Enter to keep existing value.\n", "dim"))

    def ask(prompt, default=""):
        val = input(c(f"  {prompt} [{default}]: ", "cyan")).strip()
        return val if val else default

    s["title"]       = ask("Title",       s["title"])
    s["language"]    = ask("Language",    s.get("language",""))
    s["description"] = ask("Description", s.get("description",""))
    tags_in          = ask("Tags",         ",".join(s.get("tags", [])))
    s["tags"]        = [t.strip() for t in tags_in.split(",") if t.strip()]

    rewrite = input(c("  Rewrite code? [y/N]: ", "yellow")).strip().lower()
    if rewrite == "y":
        print(c("  Enter code (type END on a new line to finish):", "cyan"))
        lines = []
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == "END":
                break
            lines.append(line)
        s["code"] = "\n".join(lines)

    snippets[idx] = s
    save(snippets)
    print(c(f"\n  ✓ Snippet #{args.id} updated.", "green"))


def interactive_mode():
    print(c("Code Snippet Manager", "bold") + "  —  store, search, copy snippets\n")
    print("Commands: add, list, show <id>, search <q>, copy <id>, edit <id>, delete <id>, quit\n")

    while True:
        try:
            line = input(c("snippet> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd   = parts[0].lower()

        class A:
            pass

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "add":
            a = A(); cmd_add(a)
        elif cmd == "list":
            a = A(); a.lang = None; a.tag = None; cmd_list(a)
        elif cmd == "show" and len(parts) > 1:
            a = A(); a.id = int(parts[1]); cmd_show(a)
        elif cmd == "search" and len(parts) > 1:
            a = A(); a.query = " ".join(parts[1:]); cmd_search(a)
        elif cmd == "copy" and len(parts) > 1:
            a = A(); a.id = int(parts[1]); cmd_copy(a)
        elif cmd == "delete" and len(parts) > 1:
            a = A(); a.id = int(parts[1]); cmd_delete(a)
        elif cmd == "edit" and len(parts) > 1:
            a = A(); a.id = int(parts[1]); cmd_edit(a)
        else:
            print(c("  Unknown command. Type 'list' to see your snippets.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="Code snippet manager")
    sub    = parser.add_subparsers(dest="command")

    sub.add_parser("add",    help="Add a new snippet")
    ls = sub.add_parser("list",   help="List all snippets")
    ls.add_argument("--lang", help="Filter by language")
    ls.add_argument("--tag",  help="Filter by tag")

    sh = sub.add_parser("show",   help="Show a snippet")
    sh.add_argument("id", type=int)

    sr = sub.add_parser("search", help="Search snippets")
    sr.add_argument("query")

    cp = sub.add_parser("copy",   help="Copy snippet to clipboard")
    cp.add_argument("id", type=int)

    ed = sub.add_parser("edit",   help="Edit a snippet")
    ed.add_argument("id", type=int)

    dl = sub.add_parser("delete", help="Delete a snippet")
    dl.add_argument("id", type=int)

    args = parser.parse_args()

    if   args.command == "add":    cmd_add(args)
    elif args.command == "list":   cmd_list(args)
    elif args.command == "show":   cmd_show(args)
    elif args.command == "search": cmd_search(args)
    elif args.command == "copy":   cmd_copy(args)
    elif args.command == "edit":   cmd_edit(args)
    elif args.command == "delete": cmd_delete(args)
    else:                          interactive_mode()


if __name__ == "__main__":
    main()
