"""Directory Tree Viewer — CLI tool.

Displays a directory tree in a format similar to the Unix `tree` command.
Supports depth limiting, filtering by extension, showing file sizes,
and exporting the tree to a text file.

Usage:
    python main.py
    python main.py /path/to/dir
"""

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_tree(
    root: Path,
    max_depth: int = 0,
    show_hidden: bool = False,
    show_size: bool = False,
    ext_filter: set[str] | None = None,
    prefix: str = "",
    depth: int = 0,
) -> list[str]:
    """Return lines of the tree as a list of strings."""
    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        lines.append(f"{prefix}[Permission Denied]")
        return lines

    entries = [
        e for e in entries
        if show_hidden or not e.name.startswith(".")
    ]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if entry.is_file():
            if ext_filter and entry.suffix.lower() not in ext_filter:
                continue
            size_str = ""
            if show_size:
                try:
                    sz = entry.stat().st_size
                    size_str = f"  [{_human(sz)}]"
                except OSError:
                    size_str = "  [?]"
            lines.append(f"{prefix}{connector}{entry.name}{size_str}")
        else:
            lines.append(f"{prefix}{connector}{entry.name}/")
            if max_depth == 0 or depth < max_depth - 1:
                sub = build_tree(
                    entry, max_depth, show_hidden, show_size,
                    ext_filter, prefix + extension, depth + 1,
                )
                lines.extend(sub)
    return lines


def _human(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}T"


def count_tree(root: Path, recursive: bool = True) -> tuple[int, int]:
    """Return (file_count, dir_count)."""
    files = dirs = 0
    try:
        for entry in (root.rglob("*") if recursive else root.iterdir()):
            if entry.is_file():
                files += 1
            elif entry.is_dir():
                dirs += 1
    except PermissionError:
        pass
    return files, dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Directory Tree Viewer
---------------------
1. View directory tree
2. View with file sizes
3. Filter by extension
4. Export tree to file
0. Quit
"""


def get_dir(prompt: str = "  Directory (blank = current): ") -> Path:
    path_str = input(prompt).strip().strip('"')
    p = Path(path_str) if path_str else Path(".")
    return p


def print_tree(root: Path, lines: list[str], show_summary: bool = True) -> None:
    print(f"\n  {root.resolve()}")
    for line in lines:
        print(f"  {line}")
    if show_summary:
        files, dirs = count_tree(root)
        print(f"\n  {dirs} directories, {files} files")


def main() -> None:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        if root.is_dir():
            lines = build_tree(root)
            print(f"{root.resolve()}")
            for line in lines:
                print(line)
            files, dirs = count_tree(root)
            print(f"\n{dirs} directories, {files} files")
            return

    print("Directory Tree Viewer")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            root = get_dir()
            if not root.is_dir():
                print(f"  Not a directory: {root}")
                continue
            depth_s = input("  Max depth (0 = unlimited): ").strip()
            depth   = int(depth_s) if depth_s.isdigit() else 0
            hidden  = input("  Show hidden files? (y/n, default n): ").strip().lower() == "y"
            lines   = build_tree(root, depth, hidden)
            print_tree(root, lines)

        elif choice == "2":
            root = get_dir()
            if not root.is_dir():
                print(f"  Not a directory: {root}")
                continue
            depth_s = input("  Max depth (0 = unlimited): ").strip()
            depth   = int(depth_s) if depth_s.isdigit() else 0
            lines   = build_tree(root, depth, show_size=True)
            print_tree(root, lines)

        elif choice == "3":
            root = get_dir()
            if not root.is_dir():
                print(f"  Not a directory: {root}")
                continue
            exts_raw = input("  Extensions (e.g. .py .txt .md): ").strip()
            ext_filter = {e if e.startswith(".") else "." + e
                          for e in exts_raw.split()} if exts_raw else None
            depth_s = input("  Max depth (0 = unlimited): ").strip()
            depth   = int(depth_s) if depth_s.isdigit() else 0
            lines   = build_tree(root, depth, ext_filter=ext_filter)
            print_tree(root, lines)

        elif choice == "4":
            root = get_dir()
            if not root.is_dir():
                print(f"  Not a directory: {root}")
                continue
            depth_s = input("  Max depth (0 = unlimited): ").strip()
            depth   = int(depth_s) if depth_s.isdigit() else 0
            lines   = build_tree(root, depth)
            out_str = input("  Output file (default tree.txt): ").strip() or "tree.txt"
            out_path = Path(out_str)
            content = str(root.resolve()) + "\n"
            content += "\n".join(lines) + "\n"
            files, dirs = count_tree(root)
            content += f"\n{dirs} directories, {files} files\n"
            out_path.write_text(content, encoding="utf-8")
            print(f"\n  Tree saved to: {out_path.resolve()}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
