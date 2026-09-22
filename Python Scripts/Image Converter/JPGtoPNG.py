"""Convert one JPG image to PNG."""

import argparse
from pathlib import Path

from convertDynamic import convert_image


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert one JPG image to PNG.')
    parser.add_argument('source', type=Path, help='Source JPG image')
    parser.add_argument('output', type=Path, help='New PNG output path')
    args = parser.parse_args()
    convert_image(args.source, 'png', args.output)
    print(f'Created {args.output}')


if __name__ == '__main__':
    main()
