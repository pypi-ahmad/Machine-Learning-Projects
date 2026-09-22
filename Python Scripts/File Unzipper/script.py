"""Preview and safely extract one ZIP archive into a new or empty directory."""

import argparse
import stat
import zipfile
from pathlib import Path


MAX_MEMBERS = 10_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3


def is_safe_member(destination: Path, member_name: str) -> bool:
    """Return whether an archive member resolves inside the extraction directory."""
    try:
        (destination.resolve() / member_name).resolve().relative_to(destination.resolve())
    except ValueError:
        return False
    return True


def total_uncompressed_size(members: list[zipfile.ZipInfo]) -> int:
    """Return the total declared uncompressed size of archive members."""
    total_size = 0
    for member in members:
        total_size += member.file_size
    return total_size


def inspect_archive(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    """Validate archive size, paths, and symlink entries before extraction."""
    members = archive.infolist()
    if len(members) > MAX_MEMBERS:
        raise ValueError(f"Archive has more than {MAX_MEMBERS:,} members.")
    total_size = total_uncompressed_size(members)
    if total_size > MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"Archive expands beyond {MAX_UNCOMPRESSED_BYTES:,} bytes.")
    for member in members:
        if not is_safe_member(destination, member.filename):
            raise ValueError(f"Unsafe archive path: {member.filename}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"Symlink entries are not supported: {member.filename}")
    return members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Existing .zip file to inspect.")
    parser.add_argument("--destination", "-d", type=Path, help="Empty output directory; defaults to ARCHIVE without .zip.")
    parser.add_argument("--extract", action="store_true", help="Request extraction after typed confirmation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.archive.resolve()
    if not archive_path.is_file() or archive_path.suffix.lower() != ".zip":
        raise SystemExit("Archive must be an existing .zip file.")
    destination = (args.destination or archive_path.with_suffix("")).resolve()
    if destination.exists() and not destination.is_dir():
        raise SystemExit(f"Destination is not a directory: {destination}")
    if destination.exists() and list(destination.iterdir()):
        raise SystemExit(f"Destination is not empty: {destination}")
    if not destination.exists() and not destination.parent.is_dir():
        raise SystemExit(f"Destination parent does not exist: {destination.parent}")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = inspect_archive(archive, destination)
            total_size = total_uncompressed_size(members)
            print(f"Archive: {archive_path}")
            print(f"Destination: {destination}")
            print(f"Members: {len(members)}")
            print(f"Uncompressed size: {total_size:,} bytes")
            if not args.extract:
                print("Preview only. Re-run with --extract to request final confirmation.")
                return
            if input("Type EXTRACT to extract this archive: ") != "EXTRACT":
                print("Cancelled. No files were extracted.")
                return
            destination.mkdir(exist_ok=True)
            for member in members:
                archive.extract(member, destination)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Cannot extract archive: {error}") from error
    print(f"Extracted {len(members)} member(s) to {destination}.")


if __name__ == "__main__":
    main()
