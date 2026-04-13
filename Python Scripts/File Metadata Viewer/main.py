"""File Metadata Viewer — CLI tool.

Displays detailed metadata for files and directories: timestamps,
permissions, owner, MIME type detection, image EXIF info (if Pillow
available), and PDF metadata (if pypdf available).

Usage:
    python main.py
    python main.py somefile.jpg
"""

import mimetypes
import os
import platform
import stat
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def permissions_str(mode: int) -> str:
    """Return rwxrwxrwx style string."""
    flags = [
        (stat.S_IRUSR, "r"), (stat.S_IWUSR, "w"), (stat.S_IXUSR, "x"),
        (stat.S_IRGRP, "r"), (stat.S_IWGRP, "w"), (stat.S_IXGRP, "x"),
        (stat.S_IROTH, "r"), (stat.S_IWOTH, "w"), (stat.S_IXOTH, "x"),
    ]
    return "".join(c if mode & flag else "-" for flag, c in flags)


def file_metadata(path: Path) -> dict:
    st = path.stat()
    mime, _ = mimetypes.guess_type(str(path))
    is_win = platform.system() == "Windows"

    meta = {
        "Path":        str(path.resolve()),
        "Name":        path.name,
        "Type":        "Directory" if path.is_dir() else "File",
        "MIME type":   mime or "unknown",
        "Size":        f"{human_size(st.st_size)}  ({st.st_size:,} bytes)",
        "Created":     fmt_time(st.st_ctime),
        "Modified":    fmt_time(st.st_mtime),
        "Accessed":    fmt_time(st.st_atime),
        "Permissions": permissions_str(st.st_mode),
        "Mode (oct)":  oct(stat.S_IMODE(st.st_mode)),
    }

    if not is_win:
        try:
            import pwd, grp
            meta["Owner"]   = pwd.getpwuid(st.st_uid).pw_name
            meta["Group"]   = grp.getgrgid(st.st_gid).gr_name
        except Exception:
            pass
    else:
        meta["Inode"]      = st.st_ino
        meta["Hard links"] = st.st_nlink

    return meta


def image_exif(path: Path) -> dict | None:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(path)
        info = {"Format": img.format, "Mode": img.mode, "Size": f"{img.size[0]}×{img.size[1]}"}
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, (str, int, float)):
                    info[str(tag)] = str(value)[:80]
        return info
    except Exception:
        return None


def pdf_meta(path: Path) -> dict | None:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        meta = {"Pages": len(reader.pages)}
        if reader.metadata:
            for k, v in reader.metadata.items():
                meta[k.lstrip("/")] = str(v)[:80]
        return meta
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
File Metadata Viewer
--------------------
1. View file/directory metadata
2. View image EXIF data
3. View PDF metadata
4. Batch view (directory)
0. Quit
"""


def print_meta(meta: dict) -> None:
    print()
    for k, v in meta.items():
        print(f"  {k:<22}: {v}")


def main() -> None:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            print_meta(file_metadata(p))
            return

    print("File Metadata Viewer")
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
            print_meta(file_metadata(p))

        elif choice == "2":
            path_str = input("  Image file: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_file():
                print(f"  File not found: {path_str}")
                continue
            exif = image_exif(p)
            if exif:
                print_meta(exif)
            else:
                print("  No EXIF data found (or Pillow not installed).")

        elif choice == "3":
            path_str = input("  PDF file: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_file():
                print(f"  File not found: {path_str}")
                continue
            meta = pdf_meta(p)
            if meta:
                print_meta(meta)
            else:
                print("  Could not read PDF metadata (pypdf not installed or invalid PDF).")

        elif choice == "4":
            path_str = input("  Directory: ").strip().strip('"')
            d = Path(path_str)
            if not d.is_dir():
                print(f"  Not a directory: {path_str}")
                continue
            pat = input("  Pattern (default *): ").strip() or "*"
            files = sorted(f for f in d.glob(pat) if f.is_file())
            if not files:
                print("  No files found.")
                continue
            print(f"\n  {'Name':<35} {'Size':>10}  {'Modified':<19}  MIME")
            print(f"  {'-'*35} {'-'*10}  {'-'*19}  {'-'*25}")
            for f in files[:40]:
                st = f.stat()
                mime, _ = mimetypes.guess_type(str(f))
                print(f"  {f.name[:35]:<35} {human_size(st.st_size):>10}"
                      f"  {fmt_time(st.st_mtime):<19}  {mime or 'unknown'}")
            if len(files) > 40:
                print(f"  ... {len(files) - 40} more")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
