"""Copy source directories into a dated-structure backup target."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def destination_for(source_root: Path, source_file: Path, target_root: Path) -> Path:
    """Keep each source under a stable top-level directory in the backup."""
    return target_root / source_root.name / source_file.relative_to(source_root)


def needs_backup(source: Path, destination: Path) -> bool:
    """Return whether a source is newer than its plain or compressed backup."""
    candidates = (destination, destination.with_suffix(destination.suffix + ".gz"))
    timestamps = [path.stat().st_mtime for path in candidates if path.exists()]
    return not timestamps or source.stat().st_mtime > max(timestamps)


def backup(source_roots: list[Path], target_root: Path, threshold: int, apply: bool) -> int:
    """Plan or apply incremental backups and return the number of affected files."""
    changed = 0
    for source_root in source_roots:
        if not source_root.is_dir():
            raise ValueError(f"Source directory not found: {source_root}")
        for source_file in source_root.rglob("*"):
            if not source_file.is_file():
                continue
            destination = destination_for(source_root, source_file, target_root)
            if not needs_backup(source_file, destination):
                continue
            compressed = source_file.stat().st_size > threshold
            output = destination.with_suffix(destination.suffix + ".gz") if compressed else destination
            print(f"{'Copy' if apply else 'Dry run'}: {source_file} -> {output}")
            if apply:
                output.parent.mkdir(parents=True, exist_ok=True)
                if compressed:
                    with source_file.open("rb") as input_file, gzip.open(output, "wb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
                else:
                    shutil.copy2(source_file, output)
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--compress", type=int, default=1_024_000)
    parser.add_argument("--apply", action="store_true", help="write backups; default is dry run")
    args = parser.parse_args()
    if args.compress < 0:
        raise SystemExit("Error: --compress must be zero or greater.")
    try:
        count = backup([path.resolve() for path in args.source], args.target.resolve(), args.compress, args.apply)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"{'Backed up' if args.apply else 'Planned'} {count} file(s).")


if __name__ == "__main__":
    main()
