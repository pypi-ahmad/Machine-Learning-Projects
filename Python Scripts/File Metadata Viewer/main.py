"""Print filesystem metadata and optional image or PDF metadata for one path."""

import argparse
import mimetypes
import platform
import stat
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS
from pypdf import PdfReader


def fmt_time(timestamp: float) -> str:
    """Render a local filesystem timestamp."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def human_size(size: int) -> str:
    """Render a byte count using binary-sized units."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def permissions_str(mode: int) -> str:
    """Return a POSIX-style rwxrwxrwx permission string."""
    flags = (
        (stat.S_IRUSR, "r"), (stat.S_IWUSR, "w"), (stat.S_IXUSR, "x"),
        (stat.S_IRGRP, "r"), (stat.S_IWGRP, "w"), (stat.S_IXGRP, "x"),
        (stat.S_IROTH, "r"), (stat.S_IWOTH, "w"), (stat.S_IXOTH, "x"),
    )
    return "".join(character if mode & flag else "-" for flag, character in flags)


def file_metadata(path: Path) -> dict[str, str]:
    """Return standard filesystem metadata without changing the target."""
    file_stat = path.stat()
    mime_type, _ = mimetypes.guess_type(str(path))
    metadata = {
        "Path": str(path.resolve()),
        "Name": path.name,
        "Type": "Directory" if path.is_dir() else "File",
        "MIME type": mime_type or "unknown",
        "Size": f"{human_size(file_stat.st_size)} ({file_stat.st_size:,} bytes)",
        "Created": fmt_time(file_stat.st_ctime),
        "Modified": fmt_time(file_stat.st_mtime),
        "Accessed": fmt_time(file_stat.st_atime),
        "Permissions": permissions_str(file_stat.st_mode),
        "Mode (oct)": oct(stat.S_IMODE(file_stat.st_mode)),
    }
    if platform.system() == "Windows":
        metadata["Hard links"] = str(file_stat.st_nlink)
    return metadata


def image_exif(path: Path) -> dict[str, str]:
    """Return basic image and scalar EXIF metadata."""
    try:
        with Image.open(path) as image:
            metadata = {
                "Format": str(image.format),
                "Mode": str(image.mode),
                "Dimensions": f"{image.width} x {image.height}",
            }
            for tag_id, value in image.getexif().items():
                if isinstance(value, (str, int, float)):
                    metadata[str(TAGS.get(tag_id, tag_id))] = str(value)[:80]
            return metadata
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot read image metadata: {error}") from error


def pdf_metadata(path: Path) -> dict[str, str]:
    """Return PDF page count and document metadata."""
    try:
        reader = PdfReader(str(path))
    except OSError as error:
        raise ValueError(f"Cannot read PDF metadata: {error}") from error
    metadata = {"Pages": str(len(reader.pages))}
    if reader.metadata:
        for key, value in reader.metadata.items():
            metadata[key.lstrip("/")] = str(value)[:80]
    return metadata


def print_metadata(metadata: dict[str, str]) -> None:
    """Print one metadata mapping in a consistent, copyable format."""
    for key, value in metadata.items():
        print(f"{key:<14}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Existing file or directory to inspect.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--exif", action="store_true", help="Also display image metadata for a file.")
    group.add_argument("--pdf", action="store_true", help="Also display PDF metadata for a file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path.expanduser()
    if not path.exists():
        raise SystemExit(f"Path does not exist: {args.path}")
    if (args.exif or args.pdf) and not path.is_file():
        raise SystemExit("--exif and --pdf require a regular file.")

    print_metadata(file_metadata(path))
    try:
        if args.exif:
            print("\nImage metadata:")
            print_metadata(image_exif(path))
        if args.pdf:
            print("\nPDF metadata:")
            print_metadata(pdf_metadata(path))
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
