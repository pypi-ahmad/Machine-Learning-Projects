"""Split a CSV or whitespace-delimited text file into fixed-row chunks."""

import argparse
from pathlib import Path

import pandas as pd


def split_file(input_path: Path, rows_per_file: int, output_dir: Path) -> int:
    """Write chunks from input_path into a new directory and return their count."""
    if input_path.suffix not in {'.csv', '.txt'}:
        raise ValueError('Input files must use .csv or .txt extensions.')
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise ValueError('Input file must exist and contain at least one row.')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {output_dir}')

    output_dir.mkdir(parents=True)
    separator = r'\s+' if input_path.suffix == '.txt' else ','
    written = 0
    for written, chunk in enumerate(
        pd.read_csv(input_path, header=None, sep=separator, chunksize=rows_per_file), start=1
    ):
        output_path = output_dir / f'split_file{written}{input_path.suffix}'
        chunk.to_csv(
            output_path,
            header=False,
            index=False,
            sep=' ' if input_path.suffix == '.txt' else ',',
        )

    if not written:
        output_dir.rmdir()
        raise ValueError('The input file contains no rows.')
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description='Split a CSV or TXT file by row count.')
    parser.add_argument('filename', type=Path, help='Input CSV or TXT file')
    parser.add_argument('rows', type=int, help='Rows per output file')
    parser.add_argument('--output-dir', type=Path, help='New directory for split files')
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('rows must be at least 1')

    output_dir = args.output_dir or args.filename.with_name(f'{args.filename.stem}_split')
    try:
        count = split_file(args.filename, args.rows, output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Created {count} file(s) in {output_dir}')


if __name__ == '__main__':
    main()
