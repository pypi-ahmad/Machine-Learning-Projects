"""File Renamer — CLI tool.

Bulk-rename files in a directory using patterns, find-replace,
numbering, date injection, case conversion, and dry-run preview.

Usage:
    python main.py
"""

import re
import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Rename strategies
# ---------------------------------------------------------------------------

def rename_preview(
    files: list[Path],
    strategy: str,
    **kwargs,
) -> list[tuple[Path, Path]]:
    """Return list of (old_path, new_path) pairs without performing renames."""
    pairs = []
    planned_names: set[Path] = set()
    source_paths = set(files)
    for i, f in enumerate(files):
        new_name = _apply_strategy(f, i, len(files), strategy, **kwargs)
        if new_name and new_name != f.name:
            target = f.parent / new_name
            if Path(new_name).name != new_name or new_name in {".", ".."}:
                raise ValueError(f"Invalid output name for {f.name}: {new_name}")
            if target in planned_names:
                raise ValueError(f"Multiple files would be renamed to {target.name}.")
            if target.exists() and target not in source_paths:
                raise ValueError(f"Refusing to overwrite existing file: {target.name}")
            planned_names.add(target)
            pairs.append((f, target))
    return pairs


def _apply_strategy(f: Path, idx: int, total: int, strategy: str, **kw) -> str:
    stem, ext = f.stem, f.suffix

    if strategy == "replace":
        find    = kw.get("find", "")
        if not find:
            return f.name
        replace = kw.get("replace", "")
        flags   = re.IGNORECASE if kw.get("ignore_case") else 0
        new_stem = re.sub(re.escape(find), replace, stem, flags=flags)
        return new_stem + ext

    elif strategy == "regex":
        pattern = kw.get("pattern", "")
        replace = kw.get("replace", "")
        try:
            new_stem = re.sub(pattern, replace, stem)
        except re.error:
            return f.name
        return new_stem + ext

    elif strategy == "prefix":
        return kw.get("prefix", "") + f.name

    elif strategy == "suffix":
        return stem + kw.get("suffix_str", "") + ext

    elif strategy == "number":
        start  = kw.get("start", 1)
        pad    = kw.get("pad", len(str(total + start)))
        sep    = kw.get("sep", "_")
        keep   = kw.get("keep", True)
        num    = str(idx + start).zfill(pad)
        return (stem + sep + num if keep else num) + ext

    elif strategy == "date":
        fmt = kw.get("fmt", "%Y%m%d")
        date_str = datetime.now().strftime(fmt)
        pos  = kw.get("pos", "prefix")
        sep  = kw.get("sep", "_")
        if pos == "prefix":
            return date_str + sep + f.name
        return stem + sep + date_str + ext

    elif strategy == "case":
        mode = kw.get("mode", "lower")
        if mode == "lower":
            return f.name.lower()
        elif mode == "upper":
            return f.name.upper()
        elif mode == "title":
            return stem.title() + ext
        elif mode == "snake":
            new = re.sub(r"[\s\-]+", "_", stem).lower()
            return new + ext
        elif mode == "spaces":
            new = stem.replace("_", " ").replace("-", " ")
            return new + ext

    elif strategy == "ext":
        new_ext = kw.get("new_ext", ext)
        if not new_ext.startswith("."):
            new_ext = "." + new_ext
        return stem + new_ext

    return f.name


def apply_renames(pairs: list[tuple[Path, Path]]) -> int:
    renamed = 0
    for old, new in pairs:
        if new.exists() and new != old:
            print(f"  SKIP (exists): {new.name}")
            continue
        old.rename(new)
        renamed += 1
    return renamed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Return files from one directory, optionally including nested folders."""
    results = []
    if recursive:
        for root, _, names in os.walk(directory):
            for name in names:
                candidate = Path(root, name)
                if candidate.is_file():
                    results.append(candidate)
    else:
        for candidate in directory.iterdir():
            if candidate.is_file():
                results.append(candidate)
    return sorted(results)


def print_preview(pairs: list[tuple[Path, Path]]) -> None:
    if not pairs:
        print("  No files would be renamed.")
        return
    print(f"\n  Preview ({len(pairs)} file(s)):")
    for old, new in pairs[:30]:
        print(f"    {old.name}  →  {new.name}")
    if len(pairs) > 30:
        print(f"  ... and {len(pairs) - 30} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
File Renamer
------------
1. Find & replace in filename
2. Regex rename
3. Add prefix / suffix
4. Sequential numbering
5. Add date prefix/suffix
6. Change case
7. Change extension
0. Quit
"""


def get_directory() -> Path | None:
    path_str = input("  Directory: ").strip().strip('"')
    p = Path(path_str)
    if not p.is_dir():
        print(f"  Not a directory: {path_str}")
        return None
    return p


def get_file_filter(directory: Path) -> list[Path]:
    recursive_str = input("  Include subdirectories? (y/n, default n): ").strip().lower()
    recursive = recursive_str == "y"
    files = list_files(directory, recursive)
    print(f"  Found {len(files)} file(s).")
    return files


def confirm_and_apply(pairs: list[tuple[Path, Path]]) -> None:
    print_preview(pairs)
    if not pairs:
        return
    ok = input("\n  Type RENAME to apply these changes: ").strip()
    if ok == "RENAME":
        n = apply_renames(pairs)
        print(f"  Renamed {n} file(s).")
    else:
        print("  Cancelled.")


def main() -> None:
    print("File Renamer")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            directory = get_directory()
            if not directory:
                continue
            files = get_file_filter(directory)
            if not files:
                continue
            find       = input("  Find    : ")
            replace    = input("  Replace : ")
            ic         = input("  Ignore case? (y/n, default n): ").strip().lower() == "y"
            pairs = rename_preview(files, "replace", find=find, replace=replace, ignore_case=ic)
            confirm_and_apply(pairs)

        elif choice == "2":
            directory = get_directory()
            if not directory:
                continue
            files = get_file_filter(directory)
            if not files:
                continue
            pattern = input("  Regex pattern: ")
            replace = input("  Replacement  : ")
            pairs = rename_preview(files, "regex", pattern=pattern, replace=replace)
            confirm_and_apply(pairs)

        elif choice == "3":
            directory = get_directory()
            if not directory:
                continue
            files = get_file_filter(directory)
            if not files:
                continue
            sub = input("  (p)refix or (s)uffix? ").strip().lower()
            if sub.startswith("p"):
                val = input("  Prefix: ")
                pairs = rename_preview(files, "prefix", prefix=val)
            else:
                val = input("  Suffix (before extension): ")
                pairs = rename_preview(files, "suffix", suffix_str=val)
            confirm_and_apply(pairs)

        elif choice == "4":
            directory = get_directory()
            if not directory:
                continue
            files = get_file_filter(directory)
            if not files:
                continue
            start_s = input("  Start number (default 1): ").strip()
            start   = int(start_s) if start_s.isdigit() else 1
            sep     = input("  Separator (default _): ").strip() or "_"
            keep    = input("  Keep original name? (y/n, default y): ").strip().lower() != "n"
            pairs   = rename_preview(files, "number", start=start, sep=sep, keep=keep)
            confirm_and_apply(pairs)

        elif choice == "5":
            directory = get_directory()
            if not directory:
                continue
            files = get_file_filter(directory)
            if not files:
                continue
            pos = input("  (p)refix or (s)uffix? (default p): ").strip().lower()
            pos = "suffix" if pos.startswith("s") else "prefix"
            fmt = input("  Date format (default %Y%m%d): ").strip() or "%Y%m%d"
            pairs = rename_preview(files, "date", pos=pos, fmt=fmt)
            confirm_and_apply(pairs)

        elif choice == "6":
            directory = get_directory()
            if not directory:
                continue
            files = get_file_filter(directory)
            if not files:
                continue
            print("  Modes: lower / upper / title / snake / spaces")
            mode = input("  Mode: ").strip().lower()
            pairs = rename_preview(files, "case", mode=mode)
            confirm_and_apply(pairs)

        elif choice == "7":
            directory = get_directory()
            if not directory:
                continue
            files = get_file_filter(directory)
            if not files:
                continue
            new_ext = input("  New extension (e.g. txt): ").strip()
            pairs = rename_preview(files, "ext", new_ext=new_ext)
            confirm_and_apply(pairs)

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
