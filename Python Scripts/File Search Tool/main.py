"""File Search Tool - CLI tool.

Search for files by name pattern, content (grep), size, date,
or extension. Displays results with metadata.

Usage:
    python main.py
"""

import re
import os
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Search engines
# ---------------------------------------------------------------------------

MAX_CONTENT_FILE_BYTES = 2 * 1024 * 1024
MAX_CONTENT_MATCHES = 500


def iter_files(root: Path, recursive: bool) -> list[Path]:
    """Return regular non-symlink files below an explicitly chosen root."""
    files = []
    if recursive:
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                candidate = Path(directory, name)
                if candidate.is_file() and not candidate.is_symlink():
                    files.append(candidate)
    else:
        for candidate in root.iterdir():
            if candidate.is_file() and not candidate.is_symlink():
                files.append(candidate)
    return sorted(files)

def search_by_name(
    root: Path,
    pattern: str,
    recursive: bool = True,
    ignore_case: bool = True,
) -> list[Path]:
    results = []
    flags = re.IGNORECASE if ignore_case else 0
    try:
        matcher = re.compile(pattern, flags)
    except re.error:
        matcher = re.compile(re.escape(pattern), flags)
    for file_path in iter_files(root, recursive):
        if matcher.search(file_path.name):
            results.append(file_path)
    return results


def search_by_content(
    root: Path,
    keyword: str,
    extensions: set[str] | None = None,
    recursive: bool = True,
    ignore_case: bool = True,
) -> list[tuple[Path, int, str]]:
    """Return (file, line_number, line) for each match."""
    matches = []
    flags = re.IGNORECASE if ignore_case else 0
    try:
        pattern = re.compile(re.escape(keyword), flags)
    except re.error:
        return matches

    for file_path in iter_files(root, recursive):
        if extensions and file_path.suffix.lower() not in extensions:
            continue
        try:
            if file_path.stat().st_size > MAX_CONTENT_FILE_BYTES:
                continue
            with file_path.open(encoding="utf-8", errors="replace") as file:
                for i, line in enumerate(file, 1):
                    if pattern.search(line):
                        matches.append((file_path, i, line.strip()[:120]))
                    if len(matches) >= MAX_CONTENT_MATCHES:
                        return matches
        except (PermissionError, OSError):
            pass
    return matches


def search_by_extension(root: Path, extensions: set[str],
                         recursive: bool = True) -> list[Path]:
    return [file_path for file_path in iter_files(root, recursive) if file_path.suffix.lower() in extensions]


def search_by_size(root: Path, min_bytes: int = 0, max_bytes: int = 0,
                    recursive: bool = True) -> list[Path]:
    results = []
    for file_path in iter_files(root, recursive):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size >= min_bytes and (max_bytes == 0 or size <= max_bytes):
            results.append(file_path)
    return sorted(results, key=lambda file_path: file_path.stat().st_size, reverse=True)


def search_by_date(root: Path, days: int, older: bool = False,
                    recursive: bool = True) -> list[Path]:
    threshold = datetime.now() - timedelta(days=days)
    results = []
    for file_path in iter_files(root, recursive):
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        except OSError:
            continue
        if older:
            if mtime < threshold:
                results.append(file_path)
        else:
            if mtime >= threshold:
                results.append(file_path)
    return sorted(results, key=lambda file_path: file_path.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def print_files(files: list[Path], max_n: int = 30) -> None:
    if not files:
        print("  No files found.")
        return
    print(f"\n  Found {len(files)} file(s):")
    for f in files[:max_n]:
        try:
            st = f.stat()
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {human_size(st.st_size):>10}  {mtime}  {f}")
        except OSError:
            print(f"  {'?':>10}  {'?':>16}  {f}")
    if len(files) > max_n:
        print(f"  ... {len(files) - max_n} more results")


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    for suffix, mul in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]:
        if s.endswith(suffix):
            try:
                return int(float(s[:-len(suffix)].strip()) * mul)
            except ValueError:
                return 0
    try:
        return int(s)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
File Search Tool
----------------
1. Search by filename (pattern/regex)
2. Search by content (grep)
3. Search by extension
4. Search by size
5. Search by date (modified)
0. Quit
"""


def get_dir() -> Path | None:
    path_str = input("  Search in: ").strip().strip('"')
    if not path_str:
        print("  A directory path is required.")
        return None
    p = Path(path_str)
    if not p.is_dir():
        print(f"  Not a directory: {p}")
        return None
    return p


def main() -> None:
    print("File Search Tool")
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
            pattern = input("  Name pattern (regex or plain): ").strip()
            if not pattern:
                continue
            rec = input("  Recursive? (y/n, default y): ").strip().lower() != "n"
            ic  = input("  Ignore case? (y/n, default y): ").strip().lower() != "n"
            results = search_by_name(root, pattern, rec, ic)
            print_files(results)

        elif choice == "2":
            root = get_dir()
            if not root:
                continue
            keyword = input("  Search keyword: ").strip()
            if not keyword:
                continue
            exts_raw = input("  Limit to extensions (e.g. .py .txt, blank=all): ").strip()
            ext_filter = {e if e.startswith(".") else "." + e
                          for e in exts_raw.split()} if exts_raw else None
            rec = input("  Recursive? (y/n, default y): ").strip().lower() != "n"
            ic  = input("  Ignore case? (y/n, default y): ").strip().lower() != "n"
            matches = search_by_content(root, keyword, ext_filter, rec, ic)
            if not matches:
                print("  No matches found.")
            else:
                print(f"\n  {len(matches)} match(es):")
                prev_file = None
                for f, lineno, line in matches[:50]:
                    if f != prev_file:
                        print(f"\n  {f}")
                        prev_file = f
                    print(f"    {lineno:>5}: {line}")
                if len(matches) > 50:
                    print(f"\n  ... {len(matches) - 50} more matches")

        elif choice == "3":
            root = get_dir()
            if not root:
                continue
            exts_raw = input("  Extensions (e.g. .py .js .txt): ").strip()
            if not exts_raw:
                continue
            ext_set = {e if e.startswith(".") else "." + e for e in exts_raw.split()}
            rec = input("  Recursive? (y/n, default y): ").strip().lower() != "n"
            results = search_by_extension(root, ext_set, rec)
            print_files(results)

        elif choice == "4":
            root = get_dir()
            if not root:
                continue
            min_s = input("  Minimum size (e.g. 1MB, 0=none): ").strip()
            max_s = input("  Maximum size (e.g. 100MB, 0=none): ").strip()
            min_b = _parse_size(min_s) if min_s and min_s != "0" else 0
            max_b = _parse_size(max_s) if max_s and max_s != "0" else 0
            rec   = input("  Recursive? (y/n, default y): ").strip().lower() != "n"
            results = search_by_size(root, min_b, max_b, rec)
            print_files(results)

        elif choice == "5":
            root = get_dir()
            if not root:
                continue
            days_s = input("  Days: ").strip()
            days = int(days_s) if days_s.isdigit() else 7
            older = input("  Older than (o) or Newer than (n)? (default n): ").strip().lower()
            is_older = older.startswith("o")
            rec = input("  Recursive? (y/n, default y): ").strip().lower() != "n"
            results = search_by_date(root, days, is_older, rec)
            label = "older" if is_older else "newer"
            print(f"\n  Files {label} than {days} days:")
            print_files(results)

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
