"""Disk Usage Analyzer — CLI tool.

Analyze disk usage by directory, file type, or age.
Shows a tree with sizes, identifies top space consumers,
and generates a text-based usage chart.

Usage:
    python main.py
    python main.py /path/to/analyze
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except PermissionError:
        pass
    return total


def top_dirs(root: Path, depth: int = 1) -> list[tuple[Path, int]]:
    """Return (subdirectory, size) sorted by size descending."""
    results = []
    try:
        for entry in root.iterdir():
            if entry.is_dir():
                sz = dir_size(entry)
                results.append((entry, sz))
    except PermissionError:
        pass
    return sorted(results, key=lambda x: x[1], reverse=True)


def ext_breakdown(root: Path) -> dict[str, int]:
    """Group total bytes by file extension."""
    from collections import defaultdict
    totals: dict[str, int] = defaultdict(int)
    for f in root.rglob("*"):
        if f.is_file():
            try:
                totals[f.suffix.lower() or "(none)"] += f.stat().st_size
            except OSError:
                pass
    return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


def old_files(root: Path, days: int = 365) -> list[tuple[Path, int, datetime]]:
    """Return files not modified in `days` days, sorted by oldest first."""
    threshold = datetime.now() - timedelta(days=days)
    results = []
    for f in root.rglob("*"):
        if f.is_file():
            try:
                st = f.stat()
                mtime = datetime.fromtimestamp(st.st_mtime)
                if mtime < threshold:
                    results.append((f, st.st_size, mtime))
            except OSError:
                pass
    return sorted(results, key=lambda x: x[2])


def large_files(root: Path, n: int = 20) -> list[tuple[Path, int]]:
    results = []
    for f in root.rglob("*"):
        if f.is_file():
            try:
                results.append((f, f.stat().st_size))
            except OSError:
                pass
    return sorted(results, key=lambda x: x[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def bar(value: int, total: int, width: int = 30) -> str:
    if not total:
        return ""
    filled = int(value / total * width)
    return "█" * filled + "░" * (width - filled)


def print_usage_chart(items: list[tuple[str, int]], title: str) -> None:
    if not items:
        return
    total = sum(v for _, v in items)
    print(f"\n  {title}  (total: {human_size(total)})")
    print(f"  {'Name':<30} {'Size':>10}  {'%':>6}  Chart")
    print(f"  {'-'*30} {'-'*10}  {'-'*6}  {'-'*30}")
    for name, sz in items[:20]:
        pct = sz / total * 100 if total else 0
        b   = bar(sz, total)
        print(f"  {str(name)[:30]:<30} {human_size(sz):>10}  {pct:>5.1f}%  {b}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Disk Usage Analyzer
-------------------
1. Top subdirectories by size
2. Extension breakdown
3. Largest files
4. Old files (not modified in N days)
5. Drive/partition usage
0. Quit
"""


def get_dir() -> Path | None:
    path_str = input("  Directory (blank = current): ").strip().strip('"')
    p = Path(path_str) if path_str else Path(".")
    if not p.is_dir():
        print(f"  Not a directory: {p}")
        return None
    return p


def main() -> None:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        if root.is_dir():
            dirs = top_dirs(root)
            print_usage_chart([(str(d.relative_to(root)), sz) for d, sz in dirs[:15]],
                              f"Subdirectories in {root}")
            return

    print("Disk Usage Analyzer")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            root = get_dir()
            if not root:
                continue
            print(f"  Calculating sizes in {root}...")
            dirs = top_dirs(root)
            items = [(str(d.relative_to(root) if root in d.parents or d == root else d.name), sz)
                     for d, sz in dirs]
            print_usage_chart(items, "Top Subdirectories")

        elif choice == "2":
            root = get_dir()
            if not root:
                continue
            print(f"  Scanning extensions in {root}...")
            breakdown = ext_breakdown(root)
            print_usage_chart(list(breakdown.items())[:20], "By File Extension")

        elif choice == "3":
            root = get_dir()
            if not root:
                continue
            n_s = input("  How many largest files? (default 20): ").strip()
            n = int(n_s) if n_s.isdigit() else 20
            print(f"  Scanning {root}...")
            files = large_files(root, n)
            if not files:
                print("  No files found.")
                continue
            total = sum(sz for _, sz in files)
            print(f"\n  Top {len(files)} large files:")
            for f, sz in files:
                print(f"  {human_size(sz):>12}  {f}")

        elif choice == "4":
            root = get_dir()
            if not root:
                continue
            days_s = input("  Not modified in N days (default 365): ").strip()
            days = int(days_s) if days_s.isdigit() else 365
            print(f"  Scanning for files older than {days} days...")
            results = old_files(root, days)
            if not results:
                print(f"  No files older than {days} days.")
                continue
            total_sz = sum(sz for _, sz, _ in results)
            print(f"\n  {len(results)} file(s), {human_size(total_sz)} total:")
            for f, sz, mtime in results[:30]:
                print(f"  {mtime.strftime('%Y-%m-%d'):<12} {human_size(sz):>10}  {f}")
            if len(results) > 30:
                print(f"  ... {len(results) - 30} more")

        elif choice == "5":
            import shutil
            paths_to_check: list[str] = []
            if sys.platform == "win32":
                import string, ctypes
                bitmask = ctypes.windll.kernel32.GetLogicalDrives()
                for letter in string.ascii_uppercase:
                    if bitmask & 1:
                        paths_to_check.append(f"{letter}:\\")
                    bitmask >>= 1
            else:
                paths_to_check = ["/"]
                try:
                    for line in os.popen("df -h").read().splitlines()[1:]:
                        parts = line.split()
                        if len(parts) >= 6:
                            paths_to_check.append(parts[5])
                except Exception:
                    pass

            print(f"\n  {'Drive':<12} {'Total':>10} {'Used':>10} {'Free':>10} {'Use%':>6}")
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
            for p in paths_to_check:
                try:
                    usage = shutil.disk_usage(p)
                    pct   = usage.used / usage.total * 100 if usage.total else 0
                    print(f"  {p:<12} {human_size(usage.total):>10}"
                          f" {human_size(usage.used):>10}"
                          f" {human_size(usage.free):>10} {pct:>5.1f}%")
                except (PermissionError, OSError):
                    pass

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
