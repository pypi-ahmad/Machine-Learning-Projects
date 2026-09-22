# Split File

> CLI tool that splits a CSV or TXT file into smaller files based on a specified row count.

## Overview

This script reads a CSV or TXT file with pandas and writes output files containing the requested number of rows. It creates a new output directory and never removes an existing one.

## Features

- Splits CSV and TXT files into smaller chunks by row count
- Automatically detects file extension (`.csv` or `.txt`) and preserves it in output files
- Streams input in row chunks instead of loading the whole file
- Refuses to overwrite an existing output directory
- Handles remainder rows — if the file doesn't divide evenly, leftover rows go into a final file
- Sequential file naming (`split_file1.csv`, `split_file2.csv`, etc.)

## Project Structure

```
Split_File/
├── split_files.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.13+
- `pandas`, managed by uv in `pyproject.toml`

## Installation

```bash
cd Split_File
uv sync
```

## Usage

```bash
uv run python split_files.py <filename> <rows>
```

**Arguments:**
| Argument | Description |
|---|---|
| `filename` | Path to the input CSV or TXT file |
| `rows` | Number of rows per output file |

**Example:**

```bash
uv run python split_files.py data.csv 100
```

Splits `data.csv` into files of 100 rows each, saved as `data_split/split_file1.csv`, `data_split/split_file2.csv`, etc.

Use `--output-dir` to choose a different new destination directory.

## How it works

1. Takes the filename and rows-per-file value from the command line.
2. Validates the `.csv` or `.txt` input and chooses a new output directory.
3. Reads input in pandas chunks of the requested size.
4. Writes each chunk as a consistently delimited numbered output file.

## Configuration

No configuration files. All parameters are provided via command-line arguments.

## Limitations

- Assumes the input file has no header row (`header=None`)
- Only supports `.txt` and `.csv` extensions
- Existing output directories are not reused; choose another `--output-dir` or move the existing output first

## Security Notes

No security concerns identified.

## License

Not specified.
