"""Text File Splitter — CLI tool.

Split large text files by line count, file size, delimiter pattern,
or into equal N parts.  Also merges split files back together.

Usage:
    python main.py
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def split_by_lines(source: Path, lines_per_file: int,
                    out_dir: Path | None = None) -> list[Path]:
    """Split into chunks of N lines each."""
    out_dir = out_dir or source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = source.stem, source.suffix

    all_lines = source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    parts = []
    part_idx = 1

    for i in range(0, len(all_lines), lines_per_file):
        chunk = all_lines[i : i + lines_per_file]
        out = out_dir / f"{stem}_part{part_idx:04d}{suffix}"
        out.write_text("".join(chunk), encoding="utf-8")
        parts.append(out)
        part_idx += 1

    return parts


def split_by_size(source: Path, max_bytes: int,
                   out_dir: Path | None = None) -> list[Path]:
    """Split so that each file is at most max_bytes."""
    out_dir = out_dir or source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = source.stem, source.suffix

    text  = source.read_text(encoding="utf-8", errors="replace")
    parts = []
    part_idx = 1
    pos = 0

    while pos < len(text):
        # Try to find a clean newline boundary near max_bytes
        end = min(pos + max_bytes, len(text))
        if end < len(text):
            newline = text.rfind("\n", pos, end + 1)
            if newline > pos:
                end = newline + 1
        chunk = text[pos:end]
        out = out_dir / f"{stem}_part{part_idx:04d}{suffix}"
        out.write_text(chunk, encoding="utf-8")
        parts.append(out)
        pos = end
        part_idx += 1

    return parts


def split_by_equal(source: Path, n: int,
                    out_dir: Path | None = None) -> list[Path]:
    """Split into N roughly equal parts by line count."""
    all_lines = source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    chunk_size = max(1, len(all_lines) // n)
    return split_by_lines(source, chunk_size, out_dir)


def split_by_delimiter(source: Path, delimiter: str,
                        out_dir: Path | None = None) -> list[Path]:
    """Split on a regex delimiter (keeps delimiter at end of segment)."""
    out_dir = out_dir or source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = source.stem, source.suffix

    text     = source.read_text(encoding="utf-8", errors="replace")
    segments = re.split(f"({re.escape(delimiter)})", text)
    # Recombine delimiter with preceding segment
    chunks = []
    buf = ""
    for seg in segments:
        buf += seg
        if seg == delimiter:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)

    parts = []
    for i, chunk in enumerate(chunks, 1):
        out = out_dir / f"{stem}_part{i:04d}{suffix}"
        out.write_text(chunk, encoding="utf-8")
        parts.append(out)
    return parts


def merge_files(files: list[Path], out: Path) -> int:
    """Concatenate files and return total bytes written."""
    with open(out, "w", encoding="utf-8") as fout:
        for f in sorted(files):
            fout.write(f.read_text(encoding="utf-8", errors="replace"))
    return out.stat().st_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


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


def print_parts(parts: list[Path]) -> None:
    print(f"\n  Created {len(parts)} file(s):")
    for p in parts:
        print(f"    {p.name:<50}  {human_size(p.stat().st_size)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Text File Splitter
------------------
1. Split by line count
2. Split by file size
3. Split into N equal parts
4. Split by delimiter
5. Merge split files
0. Quit
"""


def get_file(prompt: str) -> Path | None:
    path_str = input(prompt).strip().strip('"')
    p = Path(path_str)
    if not p.is_file():
        print(f"  File not found: {path_str}")
        return None
    return p


def get_out_dir(source: Path) -> Path:
    d = input("  Output directory (blank = same as source): ").strip().strip('"')
    return Path(d) if d else source.parent


def main() -> None:
    print("Text File Splitter")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            src = get_file("  Source file: ")
            if not src:
                continue
            n_s = input("  Lines per file: ").strip()
            n = int(n_s) if n_s.isdigit() else 1000
            out_dir = get_out_dir(src)
            parts = split_by_lines(src, n, out_dir)
            print_parts(parts)

        elif choice == "2":
            src = get_file("  Source file: ")
            if not src:
                continue
            sz_s = input("  Max size per file (e.g. 1MB, 500KB): ").strip()
            sz   = _parse_size(sz_s)
            if sz <= 0:
                print("  Invalid size.")
                continue
            out_dir = get_out_dir(src)
            parts = split_by_size(src, sz, out_dir)
            print_parts(parts)

        elif choice == "3":
            src = get_file("  Source file: ")
            if not src:
                continue
            n_s = input("  Number of parts: ").strip()
            n = int(n_s) if n_s.isdigit() and int(n_s) > 0 else 2
            out_dir = get_out_dir(src)
            parts = split_by_equal(src, n, out_dir)
            print_parts(parts)

        elif choice == "4":
            src = get_file("  Source file: ")
            if not src:
                continue
            delim = input("  Delimiter string: ")
            if not delim:
                continue
            out_dir = get_out_dir(src)
            parts = split_by_delimiter(src, delim, out_dir)
            print_parts(parts)

        elif choice == "5":
            dir_s = input("  Directory with split files: ").strip().strip('"')
            d = Path(dir_s)
            if not d.is_dir():
                print(f"  Not a directory: {dir_s}")
                continue
            pat = input("  File pattern (e.g. myfile_part*.txt): ").strip() or "*"
            files = sorted(d.glob(pat))
            files = [f for f in files if f.is_file()]
            if not files:
                print("  No files matched.")
                continue
            out_s = input("  Output file path: ").strip().strip('"')
            if not out_s:
                continue
            out = Path(out_s)
            sz = merge_files(files, out)
            print(f"\n  Merged {len(files)} file(s) → {out}  ({human_size(sz)})")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
