"""Batch Text Replacer — CLI tool.

Find and replace text across multiple files in a directory.
Supports plain text, regex, case-insensitive, dry-run preview,
and backup creation.

Usage:
    python main.py
    python main.py /path/to/dir --find "foo" --replace "bar" --ext .py .txt
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def find_replace(path: Path, pattern: str, replacement: str,
                 use_regex: bool = False, ignore_case: bool = False,
                 dry_run: bool = False, backup: bool = True) -> dict:

    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"path": path, "error": str(e), "count": 0}

    flags = re.IGNORECASE if ignore_case else 0

    if use_regex:
        try:
            new_text, count = re.subn(pattern, replacement, original, flags=flags)
        except re.error as e:
            return {"path": path, "error": f"Regex error: {e}", "count": 0}
    else:
        escaped  = re.escape(pattern)
        new_text, count = re.subn(escaped, replacement, original, flags=flags)

    if count == 0:
        return {"path": path, "count": 0}

    if not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(new_text, encoding="utf-8")

    return {"path": path, "count": count, "dry_run": dry_run}


def scan_and_replace(root: Path, pattern: str, replacement: str,
                     extensions: list[str] | None = None,
                     use_regex: bool = False, ignore_case: bool = False,
                     dry_run: bool = False, backup: bool = True) -> list[dict]:
    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and (not extensions or p.suffix.lower() in extensions)
        and p.suffix.lower() not in (".bak", ".pyc")
    ]

    results = []
    for fpath in files:
        r = find_replace(fpath, pattern, replacement,
                         use_regex, ignore_case, dry_run, backup)
        results.append(r)
    return results


def print_results(results: list[dict], root: Path, dry_run: bool):
    prefix = "[DRY RUN]" if dry_run else ""
    total  = 0
    for r in results:
        if r.get("error"):
            print(f"  ✗ {r['path'].relative_to(root)}: {r['error']}")
        elif r.get("count", 0) > 0:
            rel = r["path"].relative_to(root)
            print(f"  ✓ {prefix} {rel}: {r['count']} replacement(s)")
            total += r["count"]
    changed = sum(1 for r in results if r.get("count", 0) > 0)
    print(f"\n  {changed} file(s) changed, {total} total replacement(s).")


def main():
    parser = argparse.ArgumentParser(description="Batch Text Replacer")
    parser.add_argument("path",         nargs="?")
    parser.add_argument("--find","-f",  default=None)
    parser.add_argument("--replace","-r", default=None)
    parser.add_argument("--ext",        nargs="+")
    parser.add_argument("--regex",      action="store_true")
    parser.add_argument("--ignore-case",action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--no-backup",  action="store_true")
    args = parser.parse_args()

    if args.path and args.find is not None and args.replace is not None:
        root = Path(args.path)
        exts = [e if e.startswith(".") else "." + e for e in args.ext] if args.ext else None
        results = scan_and_replace(root, args.find, args.replace, exts,
                                    args.regex, args.ignore_case,
                                    args.dry_run, not args.no_backup)
        print_results(results, root, args.dry_run)
        return

    print("Batch Text Replacer")
    print("────────────────────────────")

    while True:
        root_str = input("\nDirectory [.]: ").strip() or "."
        root     = Path(root_str)
        if not root.is_dir():
            print(f"  Not a directory: {root}")
            continue

        find_str = input("  Find:          ").strip()
        if not find_str:
            print("  Empty search string.")
            continue
        repl_str = input("  Replace with:  ").strip()

        ext_str  = input("  File extensions (e.g. .py .txt, blank=all): ").strip()
        exts     = [e.strip() if e.strip().startswith(".") else "." + e.strip()
                    for e in ext_str.split() if e.strip()] or None

        use_re   = input("  Use regex? [y/N]: ").strip().lower() == "y"
        icase    = input("  Case-insensitive? [y/N]: ").strip().lower() == "y"
        dry      = input("  Dry run (preview only)? [Y/n]: ").strip().lower() != "n"
        backup   = input("  Create .bak backups? [Y/n]: ").strip().lower() != "n"

        results = scan_and_replace(root, find_str, repl_str, exts,
                                    use_re, icase, dry, backup)
        print_results(results, root, dry)

        if dry:
            apply = input("\n  Apply changes? [y/N]: ").strip().lower()
            if apply == "y":
                results = scan_and_replace(root, find_str, repl_str, exts,
                                            use_re, icase, False, backup)
                print_results(results, root, False)

        again = input("\n  Run another replacement? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
