# Extract Text From PDF Using Python

> A script that converts PDF files to images and extracts text using OCR (Tesseract) via `pdf2image` and `pytesseract`.

## Overview

This script scans a directory for PDF files, converts each page to a PPM image using `pdf2image`, then runs Tesseract OCR on each page image via `pytesseract` to extract text. The extracted text from each PDF is saved to a corresponding `result<N>.txt` file.

## Features

- Scans a directory for all `.pdf` files
- Converts PDF pages to PPM images using `pdf2image` (Poppler-based)
- Extracts text from each page image using **Tesseract OCR** via `pytesseract`
- Saves extracted text to numbered result files (`result0.txt`, `result1.txt`, etc.)
- Cleans up intermediate `.ppm` files after processing
- Also detects `.docx` files in the directory (though no processing is implemented for them)

## Project Structure

```
Extract Text From PDF Using Python/
├── Extract Text From PDF using Python.py
└── README.md
```

## Requirements

- Python 3.x
- `pdf2image`
- `pytesseract`
- `Pillow` (PIL)
- **Tesseract OCR** installed on the system
- **Poppler** installed on the system (required by `pdf2image`)

## Installation

```bash
cd "Extract Text From PDF Using Python"
pip install pdf2image pytesseract Pillow
```

Additionally, install system dependencies:
- **Tesseract OCR**: [Installation guide](https://github.com/tesseract-ocr/tesseract)
- **Poppler**: Required by `pdf2image` — on Windows, download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows); on Linux, `sudo apt install poppler-utils`

## Usage

1. Edit the script and set the `PATH` variable to the directory containing your PDF files.
2. Run the script:

```bash
python "Extract Text From PDF using Python.py"
```

Extracted text is saved as `result0.txt`, `result1.txt`, etc. in the same `PATH` directory.

## How It Works

1. **`delete_ppms()`** — Cleans up any existing `.ppm` or `.DS_Store` files in the `PATH` directory.
2. **File discovery** — Lists all files in `PATH`, separating `.pdf` and `.docx` files into separate lists.
3. **`pdf_extract(file, i)`**:
   - Calls `pdf2image.convert_from_path()` to convert PDF pages to PPM images in the `PATH` directory.
   - Renames generated PPM files to a sequential pattern (`image<i>-<j>.ppm`).
   - Opens each PPM image with PIL and runs `pytesseract.image_to_string()` to extract text.
   - Writes all extracted text to `result<i>.txt`.
4. Iterates over all discovered PDF files and processes each one.

## Configuration

- **`PATH`**: Hardcoded as `'Enter your path'` — **must be manually edited** to the actual directory path before running. The path should include a trailing separator.

## Limitations

- The `PATH` variable is set to the placeholder string `'Enter your path'` — the script **will not work** without editing this value.
- The path must include a trailing slash/backslash for proper file joining (uses string concatenation, not `os.path.join()`).
- `.docx` files are detected but **never processed** — the `docx_files` list is populated but unused.
- Intermediate `.ppm` files are renamed but only cleaned up at the start of each extraction, not at the end.
- Uses string concatenation for paths (`PATH + file`) instead of `os.path.join()`, which may fail across platforms.
- The variable `i` in the main loop shadows the global counter `i = 1` defined at the top.
- OCR accuracy depends on PDF rendering quality and Tesseract configuration.
- No error handling for missing Tesseract/Poppler installations.
- The `sys` module is imported but never used.

## License

Not specified.
