"""Preview or organize regular files into folders by their first character."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def destination_folder(filename: str) -> str:
    """Return the lowercase initial folder name or ``misc``."""
    return filename[0].lower() if filename and filename[0].isalpha() else "misc"


def plan_moves(directory: Path) -> list[tuple[Path, Path]]:
    """Plan safe file moves without changing the filesystem."""
    script_path = Path(__file__).resolve()
    moves: list[tuple[Path, Path]] = []
    for source in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
        if not source.is_file() or source.is_symlink() or source.resolve() == script_path:
            continue
        destination = directory / destination_folder(source.name) / source.name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {destination}"
            )
        moves.append((source, destination))
    return moves


def organize(directory: Path, apply: bool = False) -> list[tuple[Path, Path]]:
    """Return planned moves and optionally apply them after all checks pass."""
    moves = plan_moves(directory)
    if apply:
        for source, destination in moves:
            destination.parent.mkdir(exist_ok=True)
            shutil.move(str(source), str(destination))
    return moves


def print_moves(moves: list[tuple[Path, Path]], apply: bool) -> None:
    """Print the planned or completed moves."""
    if not moves:
        print("No regular files to organize.")
        return

    prefix = "" if apply else "[dry run] "
    for source, destination in moves:
        print(f"{prefix}{source.name} -> {destination.parent.name}/{destination.name}")


def main() -> None:
    """Parse command-line arguments and organize the selected directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing files to organize")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="move files after showing the same plan; omitted means dry run",
    )
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Error: not a directory: {directory}")

    try:
        moves = organize(directory, apply=args.apply)
    except FileExistsError as error:
        raise SystemExit(f"Error: {error}") from error
    print_moves(moves, args.apply)
    if not args.apply and moves:
        print("Run again with --apply to move these files.")


if __name__ == "__main__":
    main()
