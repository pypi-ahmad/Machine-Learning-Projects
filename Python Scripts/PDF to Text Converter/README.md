# PDF to Text Converter

## Overview

A command-line script that converts PDF files to plain text using the PyPDF2 library. Extracts text from all pages and writes it to a text file, with an optional default output directory.

**Type:** CLI Utility

## Features

- Converts any PDF file to a `.txt` file
- Processes all pages sequentially
- Interactive prompts for input PDF path and output text file path
- Automatic `temp/` directory creation for default output storage
- Prints extracted text to console during conversion for live preview
- Appends extracted text page-by-page to the output file

## Dependencies

No `requirements.txt` present. Dependencies inferred from imports:

| Package | Source                                          |
|---------|-------------------------------------------------|
| PyPDF2  | Third-party (`pip install PyPDF2`)              |
| os      | Python stdlib                                   |

## How It Works

1. A `temp/` directory is created if it does not already exist (used as default output location).
2. The user is prompted to enter the path to the source PDF file.
3. The user is prompted to enter the path for the output text file. If left blank, the output is saved to `temp/<pdf_filename>.txt`.
4. The PDF is opened in binary read mode and read with `PyPDF2.PdfFileReader()`.
5. The script iterates over all pages, extracts text from each page using `pageObj.extractText()`, appends it to the output file, and prints the extracted text to the console.
6. The PDF file object is closed after processing.

## Project Structure

```
convert pdf to text/
├── converter1.py     # Main conversion script
├── output.txt        # Sample output file
├── samplePdf1.pdf    # Sample input PDF
├── temp/             # Default output directory
└── README.md
```

## Setup & Installation

```bash
cd "convert pdf to text"
pip install PyPDF2
```

## How to Run

```bash
python converter1.py
```

The script will interactively prompt for:
1. **PDF file path** — Full path to the PDF to convert (use backslashes on Windows)
2. **TXT file path** — Full path for output text file (leave blank to use `temp/` directory)

## Configuration

No environment variables or config files required. All input is provided interactively at runtime.

## Testing

No formal test suite present.

## Limitations

- Uses `PdfFileReader` and `getPage()` / `extractText()` which are deprecated in PyPDF2 3.x+ (replaced by `PdfReader`, `.pages[]`, `.extract_text()`).
- Text extraction quality depends on the PDF structure — scanned PDFs (image-based) will produce empty or garbled output since no OCR is performed.
- Opens the output file in append mode (`a+`), so running the script multiple times without deleting the output file will duplicate content.
- No command-line argument support — requires interactive input, making it unsuitable for scripting or automation.
- The prompt asks for backslash-separated paths, which is Windows-specific.
- No error handling for missing or invalid PDF files.
