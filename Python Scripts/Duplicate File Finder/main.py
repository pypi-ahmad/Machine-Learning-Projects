"""Duplicate File Finder — CLI tool.

Scans a directory and finds duplicate files using fast hashing
(MD5 / SHA-256).  Groups duplicates, shows wasted space, and
optionally deletes extras (keeping the first copy).

Usage:
    python main.py
"""

import hashlib
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def file_hash(path: Path, algorithm: str = "md5", chunk: int = 65536) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def find_duplicates(
    directory: Path,
    recursive: bool = True,
    algorithm: str = "md5",
    min_size: int = 1,
) -> list[list[Path]]:
    """Return groups of files with identical content."""
    # Phase 1: group by size (fast filter)
    size_map: dict[int, list[Path]] = defaultdict(list)
    glob = directory.rglob if recursive else directory.glob
    for f in glob("*"):
        if f.is_file():
            sz = f.stat().st_size
            if sz >= min_size:
                size_map[sz].append(f)

    # Phase 2: hash files that share a size
    hash_map: dict[str, list[Path]] = defaultdict(list)
    candidates = [files for files in size_map.values() if len(files) > 1]
    total = sum(len(g) for g in candidates)
    processed = 0
    for group in candidates:
        for f in group:
            try:
                h = file_hash(f, algorithm)
                hash_map[h].append(f)
            except (PermissionError, OSError):
                pass
            processed += 1
            if processed % 50 == 0:
                print(f"  Hashed {processed}/{total} candidates...", end="\r")

    print(" " * 50, end="\r")  # clear progress line
    return [files for files in hash_map.values() if len(files) > 1]


def wasted_space(groups: list[list[Path]]) -> int:
    total = 0
    for group in groups:
        if len(group) > 1:
            try:
                sz = group[0].stat().st_size
                total += sz * (len(group) - 1)
            except OSError:
                pass
    return total


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Duplicate File Finder
---------------------
1. Find duplicates in directory
2. Find duplicates (dry-run, just report)
3. Delete duplicates (keep first copy)
0. Quit
"""


def main() -> None:
    print("Duplicate File Finder")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice in ("1", "2", "3"):
            path_str = input("  Directory to scan: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_dir():
                print(f"  Not a directory: {path_str}")
                continue

            rec = input("  Recursive? (y/n, default y): ").strip().lower()
            recursive = rec != "n"
            algo = input("  Hash algorithm (md5/sha256, default md5): ").strip().lower() or "md5"
            if algo not in ("md5", "sha256"):
                algo = "md5"

            print(f"\n  Scanning {'recursively' if recursive else 'non-recursively'}...")
            groups = find_duplicates(p, recursive, algo)

            if not groups:
                print("  No duplicates found.")
                continue

            waste = wasted_space(groups)
            print(f"\n  Found {len(groups)} duplicate group(s)")
            print(f"  Wasted space: {human_size(waste)}")

            for i, group in enumerate(groups[:20], 1):
                sz = group[0].stat().st_size
                print(f"\n  Group {i}  ({human_size(sz)} each, {len(group)} copies):")
                for j, f in enumerate(group):
                    marker = " [KEEP]" if j == 0 else " [DUP]"
                    print(f"    {f}{marker}")

            if len(groups) > 20:
                print(f"\n  ... and {len(groups) - 20} more groups")

            if choice == "3":
                confirm = input(
                    f"\n  Delete {sum(len(g)-1 for g in groups)} duplicate files? "
                    "(type YES to confirm): "
                ).strip()
                if confirm == "YES":
                    deleted, errors = 0, 0
                    for group in groups:
                        for f in group[1:]:  # keep first
                            try:
                                f.unlink()
                                deleted += 1
                            except OSError as e:
                                print(f"  Error deleting {f}: {e}")
                                errors += 1
                    print(f"\n  Deleted {deleted} file(s). Errors: {errors}")
                else:
                    print("  Cancelled.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
