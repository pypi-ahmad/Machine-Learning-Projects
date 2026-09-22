# PDF Text Extractor

OCR one selected PDF locally with Tesseract and save the extracted text as UTF-8.

## Setup

Install the Python dependencies:

```powershell
uv sync --no-config
```

This project also needs the [Tesseract OCR engine](https://github.com/tesseract-ocr/tesseract) installed separately and available on `PATH`. `uv` installs the Python wrapper, not the OCR engine itself.

## Run

```powershell
uv run --no-config python "Extract Text From PDF using Python.py" .\document.pdf
```

The default output is `document.txt` beside the input file. Choose a different output path or rendering resolution when needed:

```powershell
uv run --no-config python "Extract Text From PDF using Python.py" .\scanned.pdf --dpi 300 --output .\scanned-text.txt
```

Use `--help` to view the available options.

## Behavior and limits

- Accepts one existing PDF per run; it does not scan a directory.
- Renders pages with PyMuPDF and passes them to the local Tesseract installation without creating intermediate image files.
- Works without Poppler and does not make network requests.
- Overwrites the chosen output file if it already exists.
- OCR quality depends on the source scan and the installed Tesseract language data. This tool does not process `.docx` files.
