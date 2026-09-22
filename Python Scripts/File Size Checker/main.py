"""File Size Checker — CLI tool.

Displays sizes of files or directories, ranks by size, shows
totals, and highlights large files above a threshold.

Usage:
    python main.py
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def collect_files(directory: Path, recursive: bool) -> list[Path]:
    """Return accessible, non-symlink files from an explicitly chosen directory."""
    files = []
    if recursive:
        for root, _, names in os.walk(directory, followlinks=False):
            for name in names:
                candidate = Path(root, name)
                if candidate.is_file() and not candidate.is_symlink():
                    files.append(candidate)
    else:
        for candidate in directory.iterdir():
            if candidate.is_file() and not candidate.is_symlink():
                files.append(candidate)
    return files


def dir_size(path: Path) -> int:
    total = 0
    for file_path in collect_files(path, recursive=True):
        try:
            total += file_path.stat().st_size
        except OSError:
            continue
    return total


def scan_directory(
    directory: Path,
    recursive: bool = False,
    min_size: int = 0,
) -> list[tuple[Path, int]]:
    """Return (path, size_bytes) for all files matching criteria."""
    results = []
    for file_path in collect_files(directory, recursive):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size >= min_size:
            results.append((file_path, size))
    return sorted(results, key=lambda x: x[1], reverse=True)


def size_breakdown(directory: Path) -> list[tuple[str, int]]:
    """Return total size grouped by extension."""
    from collections import defaultdict
    ext_totals: dict[str, int] = defaultdict(int)
    for file_path in collect_files(directory, recursive=True):
        try:
            ext_totals[file_path.suffix.lower() or "(no ext)"] += file_path.stat().st_size
        except OSError:
            continue
    return sorted(ext_totals.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
File Size Checker
-----------------
1. Check file/directory size
2. Scan directory (ranked by size)
3. Find large files (above threshold)
4. Size breakdown by extension
0. Quit
"""


def main() -> None:
    print("File Size Checker")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            path_str = input("  Path: ").strip().strip('"')
            p = Path(path_str)
            if not p.exists():
                print(f"  Not found: {path_str}")
                continue
            if p.is_file():
                sz = file_size(p)
                print(f"\n  File : {p.name}")
                print(f"  Size : {human_size(sz)}  ({sz:,} bytes)")
            else:
                sz = dir_size(p)
                count = len(collect_files(p, recursive=True))
                print(f"\n  Dir  : {p.name}")
                print(f"  Files: {count:,}")
                print(f"  Size : {human_size(sz)}  ({sz:,} bytes)")

        elif choice == "2":
            path_str = input("  Directory: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_dir():
                print(f"  Not a directory: {path_str}")
                continue
            rec = input("  Recursive? (y/n, default n): ").strip().lower() == "y"
            results = scan_directory(p, rec, 0)
            if not results:
                print("  No files found.")
                continue
            total = sum(sz for _, sz in results)
            print(f"\n  {len(results)} files, total {human_size(total)}")
            print(f"\n  {'Size':>12}  File")
            print(f"  {'-'*12}  {'-'*50}")
            for fp, sz in results[:30]:
                rel = fp.relative_to(p) if rec else fp.name
                print(f"  {human_size(sz):>12}  {str(rel)[:60]}")
            if len(results) > 30:
                print(f"  ... and {len(results) - 30} more files")

        elif choice == "3":
            path_str = input("  Directory: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_dir():
                print(f"  Not a directory: {path_str}")
                continue
            thresh_str = input("  Min size (e.g. 1MB, 500KB, 10000): ").strip()
            thresh = _parse_size(thresh_str)
            results = scan_directory(p, recursive=True, min_size=thresh)
            if not results:
                print(f"  No files larger than {human_size(thresh)}.")
            else:
                print(f"\n  {len(results)} file(s) ≥ {human_size(thresh)}:")
                for fp, sz in results[:50]:
                    print(f"  {human_size(sz):>12}  {fp}")

        elif choice == "4":
            path_str = input("  Directory: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_dir():
                print(f"  Not a directory: {path_str}")
                continue
            breakdown = size_breakdown(p)
            total = sum(sz for _, sz in breakdown)
            print(f"\n  Extension breakdown (total {human_size(total)}):")
            print(f"\n  {'Ext':<12} {'Size':>10}  {'%':>6}  Bar")
            print(f"  {'-'*12} {'-'*10}  {'-'*6}  {'-'*20}")
            for ext, sz in breakdown[:20]:
                pct = sz / total * 100 if total else 0
                bar = "#" * int(pct / 5)
                print(f"  {ext:<12} {human_size(sz):>10}  {pct:>5.1f}%  {bar}")

        else:
            print("  Invalid choice.")


def _parse_size(s: str) -> int:
    """Parse '1MB', '500KB', '10000' → bytes."""
    s = s.strip().upper()
    multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "B": 1}
    for suffix, mul in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            try:
                return int(float(s[: -len(suffix)].strip()) * mul)
            except ValueError:
                return 0
    try:
        return int(s)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
