"""Create a ZIP archive from one file or a directory tree."""

import argparse
import os
import zipfile
from pathlib import Path


def default_output(source: Path) -> Path:
    """Return the archive name used when --output is not provided."""
    return source.with_name(f"{source.name}.zip")


def collect_files(source: Path, output: Path) -> list[Path]:
    """Return files to archive, excluding the destination archive itself."""
    if source.is_file():
        return [source]

    files = []
    output_path = output.resolve()
    for root, _, names in os.walk(source):
        for name in sorted(names):
            candidate = Path(root, name)
            if candidate.resolve() != output_path:
                files.append(candidate)
    return files


def create_archive(source: Path, output: Path, files: list[Path]) -> None:
    """Write the requested files using safe relative names for directories."""
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if source.is_file():
            archive.write(source, arcname=source.name)
            return
        for file_path in files:
            archive.write(file_path, arcname=file_path.relative_to(source))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File or directory to archive.")
    parser.add_argument("--output", "-o", type=Path, help="Destination ZIP path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing ZIP.")
    parser.add_argument("--dry-run", action="store_true", help="Show the planned archive without writing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file() and not source.is_dir():
        raise SystemExit(f"Source is not a regular file or directory: {args.source}")

    output = (args.output or default_output(source)).resolve()
    if output.suffix.lower() != ".zip":
        raise SystemExit("Output must use a .zip extension.")
    if not output.parent.is_dir():
        raise SystemExit(f"Output directory does not exist: {output.parent}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {output}. Re-run with --overwrite.")

    files = collect_files(source, output)
    print(f"Source: {source}")
    print(f"Archive: {output}")
    print(f"Files: {len(files)}")
    if args.dry_run:
        print("Preview only. No archive was created.")
        return

    create_archive(source, output, files)
    print("Archive created.")


if __name__ == "__main__":
    main()
