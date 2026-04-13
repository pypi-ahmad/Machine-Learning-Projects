"""File Organizer — CLI tool.

Automatically moves files from a source folder into categorized
subfolders based on file extension, creation date, or file type.
Supports dry-run preview before applying changes.

Usage:
    python main.py
"""

import shutil
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Extension-to-category mapping
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[str]] = {
    "Images":      [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
                    ".tiff", ".ico", ".heic", ".raw"],
    "Videos":      [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
                    ".m4v", ".3gp"],
    "Audio":       [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"],
    "Documents":   [".pdf", ".doc", ".docx", ".odt", ".rtf", ".txt", ".md",
                    ".tex", ".epub"],
    "Spreadsheets":[".xls", ".xlsx", ".ods", ".csv"],
    "Presentations":[".ppt", ".pptx", ".odp", ".key"],
    "Archives":    [".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz"],
    "Code":        [".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go",
                    ".rs", ".rb", ".php", ".html", ".css", ".sh", ".bat",
                    ".ps1", ".r", ".swift", ".kt"],
    "Data":        [".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                    ".sql", ".db", ".sqlite"],
    "Executables": [".exe", ".msi", ".deb", ".rpm", ".dmg", ".app"],
    "Fonts":       [".ttf", ".otf", ".woff", ".woff2"],
}

EXT_TO_CATEGORY: dict[str, str] = {
    ext: cat
    for cat, exts in CATEGORIES.items()
    for ext in exts
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def plan_by_extension(
    source: Path,
    dest_root: Path,
    recursive: bool = False,
) -> list[tuple[Path, Path]]:
    """Return (src_file, dest_file) pairs organized by extension category."""
    pairs = []
    glob = source.rglob if recursive else source.glob
    for f in glob("*"):
        if not f.is_file():
            continue
        category = EXT_TO_CATEGORY.get(f.suffix.lower(), "Other")
        dest = dest_root / category / f.name
        # Avoid collision
        dest = _unique(dest)
        pairs.append((f, dest))
    return pairs


def plan_by_date(
    source: Path,
    dest_root: Path,
    fmt: str = "%Y/%m",
    recursive: bool = False,
) -> list[tuple[Path, Path]]:
    """Organize files by modification date (year/month by default)."""
    pairs = []
    glob = source.rglob if recursive else source.glob
    for f in glob("*"):
        if not f.is_file():
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        folder = mtime.strftime(fmt)
        dest = dest_root / folder / f.name
        dest = _unique(dest)
        pairs.append((f, dest))
    return pairs


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def apply_moves(
    pairs: list[tuple[Path, Path]],
    copy: bool = False,
) -> tuple[int, int]:
    moved = errors = 0
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), dst)
            moved += 1
        except (OSError, shutil.Error) as e:
            print(f"  Error: {e}")
            errors += 1
    return moved, errors


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_plan(pairs: list[tuple[Path, Path]], source: Path, dest_root: Path) -> None:
    if not pairs:
        print("  No files to organize.")
        return
    from collections import Counter
    cats: Counter = Counter()
    for _, dst in pairs:
        cats[dst.parent.name] += 1
    print(f"\n  {len(pairs)} file(s) will be organized into {len(cats)} folder(s):")
    for cat, count in sorted(cats.items()):
        print(f"    {cat:<20} {count} file(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
File Organizer
--------------
1. Organize by file type (extension)
2. Organize by date (year/month)
3. Preview organization (dry-run)
0. Quit
"""


def get_dir(prompt: str) -> Path | None:
    path_str = input(prompt).strip().strip('"')
    p = Path(path_str)
    if not p.is_dir():
        print(f"  Not a directory: {path_str}")
        return None
    return p


def main() -> None:
    print("File Organizer")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice in ("1", "2", "3"):
            source = get_dir("  Source directory: ")
            if not source:
                continue
            dest_str = input("  Destination directory (blank = inside source): ").strip().strip('"')
            dest_root = Path(dest_str) if dest_str else source
            rec = input("  Recursive? (y/n, default n): ").strip().lower() == "y"

            if choice in ("1", "3"):
                pairs = plan_by_extension(source, dest_root, rec)
            else:
                fmt = input("  Date format (default %Y/%m): ").strip() or "%Y/%m"
                pairs = plan_by_date(source, dest_root, fmt, rec)

            print_plan(pairs, source, dest_root)

            if choice == "3" or not pairs:
                continue

            action = input("\n  (m)ove or (c)opy files? (default m): ").strip().lower()
            copy = action.startswith("c")
            confirm = input(
                f"  {'Copy' if copy else 'Move'} {len(pairs)} file(s)? (y/n): "
            ).strip().lower()
            if confirm == "y":
                moved, errors = apply_moves(pairs, copy)
                print(f"  {'Copied' if copy else 'Moved'} {moved} file(s). Errors: {errors}")
            else:
                print("  Cancelled.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
