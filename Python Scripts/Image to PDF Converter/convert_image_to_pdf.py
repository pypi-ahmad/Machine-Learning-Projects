"""Convert one image or a directory of images into a PDF."""

import argparse
import os
from pathlib import Path

import img2pdf

SUPPORTED_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def image_paths(source: Path) -> list[Path]:
    """Return supported image files in deterministic order."""
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError('Input image must be JPG, JPEG, or PNG.')
        return [source]
    if not source.is_dir():
        raise ValueError(f'Input path not found: {source}')

    images = []
    for name in sorted(os.listdir(source)):
        candidate = source / name
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            images.append(candidate)
    if not images:
        raise ValueError('No JPG, JPEG, or PNG images were found in the directory.')
    return images


def convert_to_pdf(source: Path, output_path: Path) -> None:
    """Convert supported images into a new PDF without overwriting it."""
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {output_path}')
    pdf_bytes = img2pdf.convert(image_paths(source))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as output_file:
        output_file.write(pdf_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert images to a PDF.')
    parser.add_argument('source', type=Path, help='Image file or directory')
    parser.add_argument('--output', type=Path, help='New PDF output path')
    args = parser.parse_args()
    default_name = f'{args.source.stem}.pdf'
    output_path = args.output or args.source.parent / default_name

    try:
        convert_to_pdf(args.source, output_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Created {output_path}')


if __name__ == '__main__':
    main()
