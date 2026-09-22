"""Batch-resize images with a terminal progress bar."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"})


def image_files(directory: Path) -> list[Path]:
    """Return supported image files directly inside one directory."""
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def resize_image(source: Path, destination: Path, size: tuple[int, int], overwrite: bool) -> None:
    """Resize one image to fit ``size`` while preserving its aspect ratio."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists")
    with Image.open(source) as image:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        image.save(destination)


def resize_images(directory: Path, size: tuple[int, int], output: Path, overwrite: bool) -> tuple[int, int]:
    """Resize supported files and return successful and skipped-file counts."""
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for source in tqdm(image_files(directory), desc="Resizing images", unit="image"):
        try:
            resize_image(source, output / source.name, size, overwrite)
        except (FileExistsError, UnidentifiedImageError, OSError) as error:
            print(f"Skipped {source.name}: {error}")
            skipped += 1
        else:
            completed += 1
    return completed, skipped


def main() -> None:
    """Parse resize options and process one directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing input images")
    parser.add_argument("--width", type=int, required=True, help="maximum output width")
    parser.add_argument("--height", type=int, required=True, help="maximum output height")
    parser.add_argument("--output", type=Path, help="output directory; defaults to DIRECTORY/resize")
    parser.add_argument("--overwrite", action="store_true", help="replace existing resized images")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    output = args.output or args.directory / "resize"
    try:
        completed, skipped = resize_images(args.directory, (args.width, args.height), output, args.overwrite)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Resized {completed} image(s); skipped {skipped}. Output: {output}")


if __name__ == "__main__":
    main()
