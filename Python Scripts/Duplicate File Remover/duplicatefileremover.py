"""Preview or permanently remove duplicate files in one directory."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

BLOCK_SIZE = 65_536


def file_hash(path: Path, algorithm: str) -> str:
    """Return a content hash while reading a file in fixed-size blocks."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as file:
        while block := file.read(BLOCK_SIZE):
            hasher.update(block)
    return hasher.hexdigest()


def find_duplicate_groups(directory: Path, algorithm: str) -> list[list[Path]]:
    """Return deterministic same-content groups from one directory level."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        try:
            grouped[file_hash(path, algorithm)].append(path)
        except OSError as error:
            print(f"Skip unreadable file {path}: {error}")
    return [paths for paths in grouped.values() if len(paths) > 1]


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse a directory, hash algorithm, and optional deletion request."""
    parser = argparse.ArgumentParser(description="Find non-recursive duplicate files.")
    parser.add_argument("directory", type=Path, nargs="?", default=Path("."),
                        help="directory to inspect (default: current directory)")
    parser.add_argument("--algorithm", choices=("sha256", "md5"), default="sha256",
                        help="content hash algorithm (default: sha256)")
    parser.add_argument("--delete", action="store_true",
                        help="permanently delete extras after typed confirmation")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Report duplicate groups and delete extras only after confirmation."""
    settings = parse_arguments(arguments)
    directory = settings.directory.resolve()
    if not directory.is_dir():
        print(f"Not a directory: {directory}")
        return 1

    groups = find_duplicate_groups(directory, settings.algorithm)
    if not groups:
        print("No duplicate files found.")
        return 0

    for index, group in enumerate(groups, 1):
        print(f"Group {index}: keep {group[0].name}")
        for duplicate in group[1:]:
            print(f"  duplicate: {duplicate.name}")

    duplicate_count = sum(len(group) - 1 for group in groups)
    if not settings.delete:
        print(f"Preview only: {duplicate_count} file(s) would be deleted. Add --delete to continue.")
        return 0
    if input(f"Type DELETE to permanently delete {duplicate_count} file(s): ").strip() != "DELETE":
        print("Cancelled.")
        return 0

    deleted = 0
    for group in groups:
        for duplicate in group[1:]:
            try:
                duplicate.unlink()
                deleted += 1
            except OSError as error:
                print(f"Could not delete {duplicate}: {error}")
    print(f"Deleted {deleted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
