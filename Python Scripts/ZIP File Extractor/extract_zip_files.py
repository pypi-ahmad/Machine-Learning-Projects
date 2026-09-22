"""Extract a ZIP archive into a directory named after the archive."""

import argparse
from pathlib import Path
import zipfile


def destination_for(archive: Path, output_directory: Path | None) -> Path:
    """Return the requested directory or a folder named after the archive."""
    if output_directory is not None:
        return output_directory
    return Path.cwd() / archive.stem


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive into ``destination``."""
    if archive.suffix.lower() != ".zip":
        raise ValueError("The input file must have a .zip extension.")

    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract a ZIP archive into a directory named after the archive."
    )
    parser.add_argument("-l", "--zippedfile", type=Path, required=True, help="ZIP file to extract")
    parser.add_argument("-o", "--output", type=Path, help="Destination directory")
    return parser.parse_args()


def main() -> None:
    """Validate the input archive and extract it."""
    args = parse_args()
    archive = args.zippedfile.expanduser()
    if not archive.is_file():
        raise SystemExit(f"Archive not found: {archive}")

    destination = destination_for(archive, args.output)
    try:
        extract_archive(archive, destination)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Extraction failed: {error}") from error

    print(f"Extracted to: {destination}")


if __name__ == "__main__":
    main()
