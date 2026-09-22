"""Preview or copy a folder's direct files into numbered groups.

Usage:
    uv run --no-config python split_and_copy.py SOURCE COUNT
    uv run --no-config python split_and_copy.py SOURCE COUNT --apply --confirm COPY
"""

import argparse
from pathlib import Path
from shutil import copy2


def source_files(source: Path) -> list[Path]:
    """Return direct, regular files only, ordered by name."""
    return sorted([entry for entry in source.iterdir() if entry.is_file() and not entry.is_symlink()])


def groups(files: list[Path], count: int) -> list[list[Path]]:
    """Split files into fixed-size groups."""
    return [files[index : index + count] for index in range(0, len(files), count)]


def show_plan(file_groups: list[list[Path]], destination: Path) -> None:
    print(f"Source groups: {len(file_groups)}")
    print(f"Destination: {destination}")
    for index, group in enumerate(file_groups):
        print(f"  data_{index}: {len(group)} file(s)")


def copy_groups(file_groups: list[list[Path]], destination: Path) -> None:
    destination.mkdir()
    for index, group in enumerate(file_groups):
        group_destination = destination / f"data_{index}"
        group_destination.mkdir()
        for source in group:
            copy2(source, group_destination / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split direct files into numbered folders.")
    parser.add_argument("source", type=Path, help="Folder containing files to copy")
    parser.add_argument("count", type=int, help="Maximum files per output folder")
    parser.add_argument("--output", type=Path, help="New destination folder (default: SOURCE_split)")
    parser.add_argument("--apply", action="store_true", help="Copy files after confirmation")
    parser.add_argument("--confirm", help="Type COPY to allow file copies")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")
    if args.count < 1:
        parser.error("count must be at least 1")

    destination = (args.output or source.with_name(f"{source.name}_split")).resolve()
    if destination == source or source in destination.parents:
        parser.error("output must be outside the source folder")
    if destination.exists():
        parser.error(f"output already exists: {destination}")

    files = source_files(source)
    file_groups = groups(files, args.count)
    if not file_groups:
        print("No direct regular files found.")
        return

    show_plan(file_groups, destination)
    if not args.apply:
        print("Preview only. Add --apply --confirm COPY to copy files.")
        return
    if args.confirm != "COPY":
        parser.error("--confirm COPY is required with --apply")

    copy_groups(file_groups, destination)
    print(f"Copied {len(files)} file(s).")


if __name__ == "__main__":
    main()
