"""Convert JPG and PNG files without overwriting existing images."""

import argparse
import os
from pathlib import Path

from PIL import Image

SUPPORTED_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def convert_image(source: Path, target_format: str, output_path: Path) -> None:
    """Convert one image to PNG or JPEG at a new output path."""
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {output_path}')

    with Image.open(source) as image:
        if target_format == 'jpeg':
            if 'A' in image.getbands():
                background = Image.new('RGB', image.size, 'white')
                background.paste(image, mask=image.getchannel('A'))
                image = background
            else:
                image = image.convert('RGB')
        image.save(output_path, target_format)


def convert_path(source: Path, target_format: str) -> tuple[int, int]:
    """Convert eligible images below a file or directory, returning converted/skipped counts."""
    target_suffix = '.jpg' if target_format == 'jpeg' else '.png'
    candidates = [source] if source.is_file() else []
    if source.is_dir():
        for root, _, files in os.walk(source):
            for filename in files:
                candidates.append(Path(root, filename))

    converted = 0
    skipped = 0
    for image_path in candidates:
        if image_path.suffix.lower() not in SUPPORTED_SUFFIXES or image_path.suffix.lower() == target_suffix:
            skipped += 1
            continue
        try:
            convert_image(image_path, target_format, image_path.with_suffix(target_suffix))
        except (FileExistsError, OSError) as error:
            print(f'Skipped {image_path}: {error}')
            skipped += 1
        else:
            print(f'Created {image_path.with_suffix(target_suffix)}')
            converted += 1
    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert JPG and PNG images.')
    parser.add_argument('source', type=Path, help='Image file or directory')
    parser.add_argument('--to', choices=('png', 'jpeg'), required=True, help='Target image format')
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f'Source path not found: {args.source}')

    converted, skipped = convert_path(args.source, args.to)
    print(f'Converted {converted} image(s); skipped {skipped}.')


if __name__ == '__main__':
    main()
