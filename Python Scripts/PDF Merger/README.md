# Merge PDFs

> A Python script to merge multiple PDF files using PyPDF2, supporting both appending and inserting at specific positions.

## Overview

This project provides a command-line utility that uses two methods for merging PDF documents: sequential appending and positional insertion. When executed, it merges sample PDF files using both methods and produces two output files.

## Features

- **Append merge**: Sequentially appends multiple PDFs into a single output file
- **Insert merge**: Inserts a PDF at a specified page position within another PDF
- Supports both file stream and direct file path inputs for merging

## Project Structure

```
Merge_pdfs/
├── merge_pdfs.py        # Main script with merge functions
├── requirements.txt     # Python dependencies
├── samplePdf1.pdf       # Sample input PDF
├── samplePdf2.pdf       # Sample input PDF
├── mergedPdf.pdf        # Output from append merge
└── mergedPdf1.pdf       # Output from insert merge
```

## Requirements

- Python 3.x
- PyPDF2 == 1.26.0

## Installation

```bash
cd Merge_pdfs
pip install -r requirements.txt
```

## Usage

```bash
python merge_pdfs.py
```

This runs both merge methods and produces:
- `mergedPdf.pdf` — result of appending `samplePdf1.pdf` and `samplePdf2.pdf`
- `mergedPdf1.pdf` — result of inserting `samplePdf2.pdf` at position 0 of `samplePdf1.pdf`

## How It Works

1. **`by_appending()`**: Opens `samplePdf1.pdf` as a file stream and appends `samplePdf2.pdf` by path, then writes the combined result to `mergedPdf.pdf`.
2. **`by_inserting()`**: Loads `samplePdf1.pdf`, then merges `samplePdf2.pdf` at page index 0 (before all existing pages), writing the result to `mergedPdf1.pdf`.

Both functions use `PyPDF2.PdfFileMerger` to handle the merging process.

## Configuration

No configuration required. To merge different files, modify the hardcoded filenames in `merge_pdfs.py`:
- Input files: `samplePdf1.pdf`, `samplePdf2.pdf`
- Output files: `mergedPdf.pdf`, `mergedPdf1.pdf`

## Limitations

- Input and output filenames are hardcoded — no CLI arguments supported
- Does not accept a dynamic list of PDFs to merge
- Uses `PyPDF2` 1.26.0 which is outdated; the `PdfFileMerger` class is deprecated in newer versions (use `PdfMerger` instead)
- The file stream opened in `by_appending()` is never explicitly closed
- No error handling for missing input files

## License

Not specified.
