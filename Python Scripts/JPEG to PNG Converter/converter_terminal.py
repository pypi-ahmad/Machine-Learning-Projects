"""Convert one JPEG image to PNG."""

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JPEG image to PNG.")
    parser.add_argument("input", type=Path, help="Source JPEG path.")
    parser.add_argument("output", type=Path, help="Destination PNG path.")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input image not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must differ.")
    with Image.open(args.input) as image:
        image.save(args.output, format="PNG")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
