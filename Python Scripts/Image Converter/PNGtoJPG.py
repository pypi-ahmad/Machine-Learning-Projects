"""Convert one PNG image to JPEG."""

import argparse
from pathlib import Path

from convertDynamic import convert_image


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert one PNG image to JPEG.')
    parser.add_argument('source', type=Path, help='Source PNG image')
    parser.add_argument('output', type=Path, help='New JPEG output path')
    args = parser.parse_args()
    convert_image(args.source, 'jpeg', args.output)
    print(f'Created {args.output}')


if __name__ == '__main__':
    main()
