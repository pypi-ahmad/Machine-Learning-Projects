"""Recent Files Tracker — CLI tool.

Scan a directory and list recently modified files.
Filter by age, extension, and size. Open files directly.

Usage:
    python main.py
    python main.py /path/to/scan --days 7
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def scan(root: Path, days: int = 7, extensions: list[str] | None = None,
         min_size: int = 0, max_results: int = 50) -> list[dict]:
    cutoff = datetime.now() - timedelta(days=days)
    results = []

    try:
        for fpath in root.rglob("*"):
            if not fpath.is_file():
                continue
            if extensions and fpath.suffix.lower() not in extensions:
                continue
            try:
                stat = fpath.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                if mtime < cutoff:
                    continue
                if stat.st_size < min_size:
                    continue
                results.append({
                    "path":  fpath,
                    "mtime": mtime,
                    "size":  stat.st_size,
                    "ext":   fpath.suffix.lower(),
                })
            except OSError:
                pass
    except PermissionError:
        pass

    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results[:max_results]


def open_file(path: Path):
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception as e:
        print(f"  Could not open: {e}")


def display(results: list[dict], root: Path):
    if not results:
        print("  No recent files found.")
        return
    print(f"\n  {'#':>3}  {'Modified':>16}  {'Size':>8}  File")
    print("  " + "─" * 70)
    for i, r in enumerate(results, 1):
        rel   = r["path"].relative_to(root)
        mtime = r["mtime"].strftime("%Y-%m-%d %H:%M")
        size  = human_size(r["size"])
        print(f"  {i:>3}  {mtime}  {size:>8}  {rel}")


def main():
    parser = argparse.ArgumentParser(description="Recent Files Tracker")
    parser.add_argument("path",       nargs="?", default=None)
    parser.add_argument("--days","-d",type=int, default=7)
    parser.add_argument("--ext",      nargs="+", help="Extensions e.g. .py .txt")
    parser.add_argument("--top",      type=int, default=50)
    args = parser.parse_args()

    if args.path:
        root = Path(args.path)
        exts = [e if e.startswith(".") else "." + e for e in args.ext] if args.ext else None
        results = scan(root, args.days, exts, max_results=args.top)
        display(results, root)
        return

    print("Recent Files Tracker")
    print("────────────────────────────")

    while True:
        root_str = input("\nDirectory to scan [.]: ").strip() or "."
        root = Path(root_str)
        if not root.is_dir():
            print(f"  Not a directory: {root}")
            continue

        days_str = input("  Days back [7]: ").strip()
        days     = int(days_str) if days_str.isdigit() else 7

        ext_str  = input("  Filter by extension (e.g. .py .txt, blank=all): ").strip()
        exts     = [e.strip() if e.strip().startswith(".") else "." + e.strip()
                    for e in ext_str.split() if e.strip()] or None

        results  = scan(root, days, exts)
        display(results, root)

        if results:
            while True:
                action = input("\n  Enter # to open, 'r' to re-scan, 'q' to quit: ").strip().lower()
                if action == "q":
                    sys.exit()
                elif action == "r":
                    break
                elif action.isdigit():
                    idx = int(action) - 1
                    if 0 <= idx < len(results):
                        open_file(results[idx]["path"])
                    else:
                        print("  Invalid number.")
                else:
                    print("  Invalid.")
        else:
            cont = input("  Scan another directory? [y/N]: ").strip().lower()
            if cont != "y":
                break


if __name__ == "__main__":
    main()
