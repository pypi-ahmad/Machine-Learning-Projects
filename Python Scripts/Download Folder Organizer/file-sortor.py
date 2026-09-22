"""Preview or organize files into folders based on their extensions."""

import argparse
import shutil
from pathlib import Path

CATEGORY_EXTENSIONS = {
    "images": {".jpg", ".png", ".jpeg", ".gif"},
    "videos": {".mp4", ".mkv"},
    "music": {".mp3", ".wav"},
    "archives": {".zip", ".tgz", ".rar", ".tar"},
    "documents": {".pdf", ".docx", ".csv", ".xlsx", ".pptx", ".doc", ".ppt", ".xls"},
    "installers": {".msi", ".exe"},
    "programs": {".py", ".c", ".cpp", ".php"},
    "design": {".xd", ".psd"},
}


def category_for(path: Path) -> str:
    """Return the configured category for a file, or ``others``."""
    suffix = path.suffix.lower()
    for category, extensions in CATEGORY_EXTENSIONS.items():
        if suffix in extensions:
            return category
    return "others"


def planned_moves(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    """Return non-conflicting file moves without modifying the filesystem."""
    moves = []
    for item in source.iterdir():
        if not item.is_file() or item.name.startswith("."):
            continue
        target = destination / category_for(item) / item.name
        if target.exists():
            print(f"Skip existing destination: {target}")
            continue
        moves.append((item, target))
    return moves


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit source and destination directories."""
    parser = argparse.ArgumentParser(description="Organize files by extension.")
    parser.add_argument("source", type=Path, help="directory containing files to organize")
    parser.add_argument("destination", type=Path, help="directory that will receive category folders")
    parser.add_argument("--execute", action="store_true", help="move files instead of previewing")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Preview file moves by default; execute only when explicitly requested."""
    settings = parse_arguments(arguments)
    source = settings.source.resolve()
    destination = settings.destination.resolve()
    if not source.is_dir():
        print(f"Source is not a directory: {source}")
        return 1
    if source == destination:
        print("Source and destination must be different directories.")
        return 1

    moves = planned_moves(source, destination)
    if not moves:
        print("No files to organize.")
        return 0
    for item, target in moves:
        print(f"{item.name} -> {target}")
    if not settings.execute:
        print(f"Preview only: {len(moves)} file(s). Add --execute to move them.")
        return 0

    for item, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(target))
    print(f"Moved {len(moves)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
