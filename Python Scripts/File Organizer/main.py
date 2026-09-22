"""Preview file organization by extension or modification date before applying it."""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path


CATEGORIES: dict[str, list[str]] = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".tiff", ".ico", ".heic", ".raw"],
    "Videos": [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".3gp"],
    "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"],
    "Documents": [".pdf", ".doc", ".docx", ".odt", ".rtf", ".txt", ".md", ".tex", ".epub"],
    "Spreadsheets": [".xls", ".xlsx", ".ods", ".csv"],
    "Presentations": [".ppt", ".pptx", ".odp", ".key"],
    "Archives": [".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz"],
    "Code": [".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css", ".sh", ".bat", ".ps1", ".r", ".swift", ".kt"],
    "Data": [".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".db", ".sqlite"],
    "Executables": [".exe", ".msi", ".deb", ".rpm", ".dmg", ".app"],
    "Fonts": [".ttf", ".otf", ".woff", ".woff2"],
}
EXT_TO_CATEGORY = {extension: category for category, extensions in CATEGORIES.items() for extension in extensions}


def is_within(path: Path, parent: Path) -> bool:
    """Return whether path resolves inside parent."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def source_files(source: Path, destination: Path, recursive: bool) -> list[Path]:
    """Collect files while excluding a distinct destination tree."""
    files = []
    if recursive:
        for directory, _, names in os.walk(source):
            for name in names:
                candidate = Path(directory, name)
                if destination != source and is_within(candidate, destination):
                    continue
                if candidate.is_file():
                    files.append(candidate)
    else:
        for candidate in source.iterdir():
            if candidate.is_file():
                files.append(candidate)
    return sorted(files)


def unique_destination(destination: Path, reserved: set[Path]) -> Path:
    """Return an unused output path, accounting for both disk and this plan."""
    if destination not in reserved and not destination.exists():
        reserved.add(destination)
        return destination
    number = 1
    while True:
        candidate = destination.with_name(f"{destination.stem}_{number}{destination.suffix}")
        if candidate not in reserved and not candidate.exists():
            reserved.add(candidate)
            return candidate
        number += 1


def plan(source: Path, destination: Path, strategy: str, recursive: bool, date_format: str) -> list[tuple[Path, Path]]:
    """Return a collision-free, non-mutating organization plan."""
    pairs = []
    reserved: set[Path] = set()
    for file_path in source_files(source, destination, recursive):
        if strategy == "extension":
            folder = EXT_TO_CATEGORY.get(file_path.suffix.lower(), "Other")
        else:
            folder = datetime.fromtimestamp(file_path.stat().st_mtime).strftime(date_format)
        target = unique_destination(destination / folder / file_path.name, reserved)
        pairs.append((file_path, target))
    return pairs


def print_plan(pairs: list[tuple[Path, Path]]) -> None:
    """Print the full planned source-to-destination mapping."""
    print(f"Planned files: {len(pairs)}")
    for source, destination in pairs:
        print(f"  {source} -> {destination}")


def apply_plan(pairs: list[tuple[Path, Path]], copy_files: bool) -> None:
    """Apply a reviewed plan, creating output folders only at this stage."""
    for source, destination in pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_files:
            shutil.copy2(source, destination)
        else:
            shutil.move(str(source), destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing directory whose files will be organized.")
    parser.add_argument("--destination", "-d", type=Path, help="Output directory; defaults to SOURCE.")
    parser.add_argument("--strategy", choices=("extension", "date"), default="extension")
    parser.add_argument("--date-format", default="%Y/%m", help="strftime folder format for --strategy date.")
    parser.add_argument("--recursive", action="store_true", help="Include nested files when the destination is outside SOURCE.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them when applying.")
    parser.add_argument("--apply", action="store_true", help="Request an explicit confirmation to perform the plan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = (args.destination or source).resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if args.recursive and destination == source:
        raise SystemExit("--recursive requires --destination outside the source directory.")
    if destination.exists() and not destination.is_dir():
        raise SystemExit(f"Destination is not a directory: {destination}")
    if not destination.exists() and not destination.parent.is_dir():
        raise SystemExit(f"Destination parent does not exist: {destination.parent}")

    pairs = plan(source, destination, args.strategy, args.recursive, args.date_format)
    print_plan(pairs)
    if not args.apply:
        print("Preview only. Re-run with --apply to request final confirmation.")
        return
    action = "COPY" if args.copy else "MOVE"
    if input(f"Type {action} to apply this plan: ") != action:
        print("Cancelled. No files were changed.")
        return
    apply_plan(pairs, args.copy)
    print(f"Applied {len(pairs)} file operation(s).")


if __name__ == "__main__":
    main()
