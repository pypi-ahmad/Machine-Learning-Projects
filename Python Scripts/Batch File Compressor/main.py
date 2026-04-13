"""Batch File Compressor — CLI tool.

Compress files or directories into ZIP, TAR.GZ, or TAR.BZ2 archives.
Also extract archives, list archive contents, and batch compress
multiple files at once.

Usage:
    python main.py
"""

import os
import tarfile
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def compress_zip(sources: list[Path], out: Path,
                  level: int = 6, recursive: bool = True) -> int:
    """Create a ZIP archive. Returns number of files added."""
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        for src in sources:
            if src.is_file():
                zf.write(src, src.name)
                count += 1
            elif src.is_dir():
                glob = src.rglob("*") if recursive else src.iterdir()
                for f in glob:
                    if f.is_file():
                        zf.write(f, f.relative_to(src.parent))
                        count += 1
    return count


def compress_tar(sources: list[Path], out: Path,
                  mode: str = "gz", recursive: bool = True) -> int:
    """Create a TAR archive (gz or bz2). Returns files added."""
    count = 0
    with tarfile.open(out, f"w:{mode}") as tf:
        for src in sources:
            tf.add(src, arcname=src.name, recursive=recursive)
            if src.is_file():
                count += 1
            else:
                count += sum(1 for _ in src.rglob("*") if _.is_file())
    return count


def extract_zip(archive: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)
        return zf.namelist()


def extract_tar(archive: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        names = tf.getnames()
        tf.extractall(out_dir)
        return names


def list_zip(archive: Path) -> list[tuple[str, int]]:
    with zipfile.ZipFile(archive, "r") as zf:
        return [(zi.filename, zi.file_size) for zi in zf.infolist()]


def list_tar(archive: Path) -> list[tuple[str, int]]:
    with tarfile.open(archive) as tf:
        return [(m.name, m.size) for m in tf.getmembers() if m.isfile()]


def detect_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if name.endswith((".tar.bz2", ".tbz2")):
        return "tar.bz2"
    if name.endswith(".tar"):
        return "tar"
    return "unknown"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def get_sources() -> list[Path]:
    print("  Enter file/directory paths (one per line, blank to finish):")
    sources = []
    while True:
        s = input("  > ").strip().strip('"')
        if not s:
            break
        p = Path(s)
        if p.exists():
            sources.append(p)
        else:
            print(f"    Not found: {s}")
    return sources


def ratio(original: int, compressed: int) -> str:
    if original == 0:
        return "N/A"
    r = (1 - compressed / original) * 100
    return f"{r:.1f}%"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Batch File Compressor
---------------------
1. Compress to ZIP
2. Compress to TAR.GZ
3. Compress to TAR.BZ2
4. Extract archive
5. List archive contents
0. Quit
"""


def main() -> None:
    print("Batch File Compressor")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice in ("1", "2", "3"):
            sources = get_sources()
            if not sources:
                continue
            out_s = input("  Output archive path: ").strip().strip('"')
            if not out_s:
                continue
            out = Path(out_s)
            rec = input("  Recursive? (y/n, default y): ").strip().lower() != "n"

            orig_size = sum(
                (f.stat().st_size for s in sources
                 for f in (s.rglob("*") if s.is_dir() else [s]) if f.is_file()),
                0,
            )

            try:
                if choice == "1":
                    lvl_s = input("  Compression level 0-9 (default 6): ").strip()
                    lvl = int(lvl_s) if lvl_s.isdigit() and 0 <= int(lvl_s) <= 9 else 6
                    count = compress_zip(sources, out, lvl, rec)
                elif choice == "2":
                    count = compress_tar(sources, out, "gz", rec)
                else:
                    count = compress_tar(sources, out, "bz2", rec)

                comp_size = out.stat().st_size
                print(f"\n  Created : {out}")
                print(f"  Files   : {count}")
                print(f"  Original: {human_size(orig_size)}")
                print(f"  Archive : {human_size(comp_size)}")
                print(f"  Ratio   : {ratio(orig_size, comp_size)} reduction")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "4":
            arch_s = input("  Archive path: ").strip().strip('"')
            arch = Path(arch_s)
            if not arch.is_file():
                print(f"  File not found: {arch_s}")
                continue
            out_s = input("  Extract to (blank = archive directory): ").strip().strip('"')
            out_dir = Path(out_s) if out_s else arch.parent / arch.stem.replace(".tar", "")
            fmt = detect_format(arch)
            try:
                if fmt == "zip":
                    names = extract_zip(arch, out_dir)
                elif fmt in ("tar.gz", "tar.bz2", "tar"):
                    names = extract_tar(arch, out_dir)
                else:
                    print(f"  Unknown format: {arch.suffix}")
                    continue
                print(f"\n  Extracted {len(names)} item(s) to {out_dir}")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "5":
            arch_s = input("  Archive path: ").strip().strip('"')
            arch = Path(arch_s)
            if not arch.is_file():
                print(f"  File not found: {arch_s}")
                continue
            fmt = detect_format(arch)
            try:
                if fmt == "zip":
                    items = list_zip(arch)
                elif fmt in ("tar.gz", "tar.bz2", "tar"):
                    items = list_tar(arch)
                else:
                    print(f"  Unknown format.")
                    continue
                total = sum(sz for _, sz in items)
                print(f"\n  {len(items)} file(s), {human_size(total)} uncompressed:")
                for name, sz in items[:40]:
                    print(f"  {human_size(sz):>10}  {name}")
                if len(items) > 40:
                    print(f"  ... {len(items) - 40} more")
            except Exception as e:
                print(f"  Error: {e}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
