# Split File

> CLI tool that splits a CSV or TXT file into multiple smaller files based on a specified row count.

## Overview

This script reads a CSV or TXT file using pandas and splits it into multiple output files, each containing a specified number of rows. Output files are written to a `file_split` directory that is created (or recreated) on each run.

## Features

- Splits CSV and TXT files into smaller chunks by row count
- Automatically detects file extension (`.csv` or `.txt`) and preserves it in output files
- Creates a clean `file_split` output directory (removes existing one if present)
- Handles remainder rows — if the file doesn't divide evenly, leftover rows go into a final file
- Sequential file naming (`split_file1.csv`, `split_file2.csv`, etc.)

## Project Structure

```
Split_File/
├── split_files.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `pandas==1.1.0` (specified in requirements.txt)

## Installation

```bash
cd Split_File
pip install -r requirements.txt
```

## Usage

```bash
python split_files.py <filename> <split_number>
```

**Arguments:**
| Argument | Description |
|---|---|
| `filename` | Path to the input CSV or TXT file |
| `split_number` | Number of rows per output file |

**Example:**

```bash
python split_files.py data.csv 100
```

Splits `data.csv` into files of 100 rows each, saved as `file_split/split_file1.csv`, `file_split/split_file2.csv`, etc.

## How It Works

1. Takes the filename and split count from `sys.argv`
2. Creates a `Split_Files` class instance that:
   - Removes and recreates the `file_split` output directory
   - Detects the file extension (`.txt` or `.csv`)
3. Reads the entire file with `pd.read_csv()` (header=None)
4. Iterates row by row, appending to a temporary DataFrame
5. Every `split_number` rows, writes the accumulated rows to a numbered output file
6. After the loop, writes any remaining rows to a final file
7. TXT files use space-separated output; CSV files use comma-separated output (except the final remainder file, which always uses comma-separated output regardless of extension)

## Configuration

No configuration files. All parameters are provided via command-line arguments.

## Limitations

- Uses the deprecated `DataFrame.append()` method — will not work on pandas 2.0+
- Reads the entire file into memory before splitting, which may be problematic for very large files
- The row-by-row append approach is inefficient; slicing the DataFrame would be faster
- Assumes the input file has no header row (`header=None`)
- Only supports `.txt` and `.csv` extensions; any non-`.txt` file is treated as CSV
- The `file_split` directory is always deleted and recreated, destroying any previous output
- No validation of command-line arguments (missing args cause an IndexError)

## Security Notes

No security concerns identified.

## License

Not specified.
