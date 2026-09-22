# Merge CSV Files

> A Python script that merges all CSV files in the current directory into a single combined CSV file.

## Overview

This script uses `glob` to find all `.csv` files in the current working directory, reads each one into a Pandas DataFrame, concatenates them all together, and writes the combined result to `combined_csv.csv`.

## Features

- Automatically discovers all `.csv` files in the current directory using glob
- Reads and concatenates multiple CSV files into a single DataFrame
- Exports the merged result to `combined_csv.csv` with UTF-8 BOM encoding (`utf-8-sig`)
- Preserves all columns across source files
- Omits the DataFrame index from the output file

## Project Structure

```
Merge_csv_files/
├── merge_csv_files.py
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.13+
- `pandas`, managed by uv in `pyproject.toml`

## Installation

```bash
cd "Merge_csv_files"
uv sync
```

## Usage

1. Place all the CSV files you want to merge in the same directory as `merge_csv_files.py`.
2. Run the script:

```bash
uv run python merge_csv_files.py
```

3. The merged output will be saved as `combined_csv.csv` in the same directory.

`combined_csv.csv` is excluded from later runs, so rerunning the script does not merge the previous output back into itself.

## How It Works

1. Uses `glob.glob('*.csv')` to find all files matching the `.csv` extension in the current directory.
2. Reads each CSV file into a Pandas DataFrame with `pd.read_csv(f)`.
3. Concatenates all DataFrames using `pd.concat()`.
4. Writes the combined DataFrame to `combined_csv.csv` with `index=False` and `encoding='utf-8-sig'` (UTF-8 with BOM, useful for Excel compatibility).

## Configuration

- **File extension:** Hardcoded to `csv` in the glob pattern.
- **Output filename:** Hardcoded as `combined_csv.csv`.
- **Output encoding:** `utf-8-sig` (UTF-8 with BOM).

## Limitations

- Only discovers CSV files in the current working directory — does not search subdirectories.
- All CSVs must have compatible column structures for a meaningful merge; mismatched columns will result in `NaN` values.
- No error handling for malformed CSV files, empty files, or encoding issues.
- No option to specify input files, output filename, or directory via CLI arguments.

## Security Notes

No security concerns identified.

## License

Not specified.
