"""Folder Size Analyzer — CLI tool.

Show the sizes of all subdirectories in a given path,
ranked largest first. Also shows extension breakdown
and largest individual files.

Usage:
    python main.py
    python main.py /path/to/folder
    python main.py . --top 20
"""

import argparse
import os
import sys
from pathlib import Path


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def scan_tree(
    path: Path,
    extension_sizes: dict[str, int],
    extension_counts: dict[str, int],
    files: list[tuple[Path, int]],
    top_directories: list[tuple[str, int]] | None = None,
) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                try:
                    size = entry.stat(follow_symlinks=False).st_size
                    file_path = Path(entry.path)
                    extension = file_path.suffix.lower() or "(none)"
                    total += size
                    extension_sizes[extension] = extension_sizes.get(extension, 0) + size
                    extension_counts[extension] = extension_counts.get(extension, 0) + 1
                    files.append((file_path, size))
                except OSError:
                    pass
            elif entry.is_dir(follow_symlinks=False):
                size = scan_tree(Path(entry.path), extension_sizes, extension_counts, files)
                total += size
                if top_directories is not None:
                    top_directories.append((entry.name, size))
    except OSError:
        pass
    return total


def bar(frac: float, width: int = 20) -> str:
    filled = int(frac * width)
    return "#" * filled + "." * (width - filled)


def analyze(root: Path, top_n: int = 15) -> None:
    print(f"\n  Analyzing: {root}")
    print("  " + "-" * 53)

    # Subdirectory sizes
    entries: list[tuple[str, int]] = []
    extension_sizes: dict[str, int] = {}
    extension_counts: dict[str, int] = {}
    files: list[tuple[Path, int]] = []
    root_size = scan_tree(root, extension_sizes, extension_counts, files, entries)
    entries.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Subdirectories (top {min(top_n, len(entries))}):")
    print(f"  {'Name':<35} {'Size':>10}  {'%':>5}  Bar")
    print("  " + "-" * 65)
    for name, size in entries[:top_n]:
        pct   = size / root_size * 100 if root_size else 0
        b     = bar(pct / 100)
        print(f"  {name:<35} {human_size(size):>10}  {pct:5.1f}%  {b}")

    top_ext = sorted(extension_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top extensions by size:")
    print(f"  {'Ext':<15} {'Count':>8} {'Size':>12}")
    print("  " + "-" * 38)
    for ext, size in top_ext:
        print(f"  {ext:<15} {extension_counts[ext]:>8,} {human_size(size):>12}")

    # Largest files
    files.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Largest files (top 10):")
    print(f"  {'File':<50} {'Size':>10}")
    print("  " + "-" * 62)
    for fpath, size in files[:10]:
        rel = str(fpath.relative_to(root))
        print(f"  {rel[:50]:<50} {human_size(size):>10}")

    print(f"\n  Total size: {human_size(root_size)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Folder Size Analyzer")
    parser.add_argument("path",  nargs="?", default=None)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")

    if args.path:
        root = Path(args.path)
        if not root.is_dir():
            print(f"Not a directory: {root}")
            sys.exit(1)
        analyze(root, args.top)
        return

    while True:
        raw = input("Path to analyze (or 'q' to quit) [.]: ").strip()
        if raw.lower() == "q":
            break
        root = Path(raw or ".")
        if root.is_dir():
            analyze(root)
        else:
            print(f"  Not a directory: {root}")


if __name__ == "__main__":
    main()
