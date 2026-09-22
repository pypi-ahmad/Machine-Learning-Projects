"""Reduce an image's dimensions and save the resized copy."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def resize_image(input_path: Path, output_path: Path, scale: float, quality: int) -> tuple[int, int]:
    """Resize an image by ``scale`` and return the new width and height."""
    if scale <= 0:
        raise ValueError("Scale must be greater than zero.")
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")

    with Image.open(input_path) as image:
        width = max(1, round(image.width / scale))
        height = max(1, round(image.height / scale))
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        save_options = {"quality": quality, "optimize": True} if output_path.suffix.lower() in {".jpg", ".jpeg"} else {}
        resized.save(output_path, **save_options)
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce an image's dimensions and file size.")
    parser.add_argument("input", type=Path, help="Source image path.")
    parser.add_argument("output", type=Path, help="Destination image path.")
    parser.add_argument("--scale", type=float, default=5, help="Dimension divisor (default: 5).")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality, from 1 to 100 (default: 85).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input image not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Choose a different output path so the source image is not overwritten.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        width, height = resize_image(args.input, args.output, args.scale, args.quality)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not resize the image: {error}") from error
    print(f"Saved {args.output} at {width}x{height} pixels.")


if __name__ == "__main__":
    main()
