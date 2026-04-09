# PDF to CSV Converter

> A Python script that converts tables in PDF files to CSV format using the tabula-py library.

## Overview

This script reads tabular data from a PDF file and converts it into a CSV file. It uses `tabula-py` (a Python wrapper for tabula-java) to extract tables from all pages of the PDF. The script includes automatic file detection and creates a blank CSV if one doesn't exist.

## Features

- Converts tables from PDF to CSV format
- Processes **all pages** of the PDF
- Auto-detects PDF files in the current directory if the default file isn't found
- Creates a blank CSV output file if it doesn't already exist
- Verbose console output showing each step of the conversion process

## Project Structure

```
PDF-To-CSV-Converter/
├── main.py               # Main conversion script
├── requirements.txt.txt  # Dependencies (note: double .txt extension)
├── sample1.csv           # Sample output CSV
└── sample1.pdf           # Sample input PDF
```

## Requirements

- Python 3.x
- `tabula-py`
- `os` (standard library)
- **Java Runtime Environment (JRE)** — required by tabula-java under the hood

## Installation

```bash
cd "PDF-To-CSV-Converter"
pip install tabula-py
```

**Note**: `tabula-py` requires Java to be installed. Install from [java.com](https://www.java.com/en/download/) or use OpenJDK.

## Usage

```bash
python main.py
```

The script will:

1. Look for `sample1.pdf` in the current directory.
2. If not found, search for any single `.pdf` file in the directory.
3. Convert the PDF tables to `sample1.csv` (or create it if it doesn't exist).
4. Print the output file path on completion.

## How It Works

1. **`pdf_csv()` function**:
   - Sets default filenames: `sample1.pdf` (input) and `sample1.csv` (output).
   - Determines the current working directory for file paths.
   - Checks if the default PDF exists; if not, scans the directory for other `.pdf` files.
   - If exactly **one** PDF is found, it uses that file; if multiple are found, it exits.
   - Checks for (or creates) the output CSV file.
   - Calls `tabula.convert_into()` with `output_format="csv"` and `pages="all"`.
2. The function runs automatically when the script is executed.

## Configuration

- **Default input filename**: `sample1.pdf` (hardcoded in `pdf_csv()`).
- **Default output filename**: `sample1.csv` (hardcoded in `pdf_csv()`).
- **Working directory**: Uses the current working directory (`os.getcwd()`).

## Limitations

- Filenames are hardcoded to `sample1.pdf` / `sample1.csv` — no CLI argument support.
- If more than one PDF exists and the default isn't found, the script exits without conversion.
- No support for selecting specific pages or table regions.
- Requires Java JRE to be installed (dependency of tabula-java).
- The requirements file has a double extension (`requirements.txt.txt`).
- No handling for PDFs without tables.
- Verbose print-based logging cannot be silenced.

## Security Notes

No security concerns identified.

## License

Not specified.
