# PDF Reader with Voice

Extract text from a local PDF or image, save it as UTF-8 text, and optionally read it through the local text-to-speech engine.

## Setup

```powershell
uv sync --no-config
```

Image OCR and scanned-PDF OCR require Tesseract installed and available on `PATH`. The project installs the Python wrapper, Pillow, PyMuPDF, and pyttsx3; it does not install the Tesseract executable or language data.

## Run

Extract embedded text from a PDF without speech:

```powershell
uv run --no-config python main.py .\document.pdf --no-speak
```

OCR a scanned PDF or image:

```powershell
uv run --no-config python main.py .\scanned.pdf --ocr --no-speak
uv run --no-config python main.py .\image.jpg --no-speak --output .\image-text.txt
```

Without `--no-speak`, the extracted text is read through the configured local speech engine.

## Behavior and limits

- `main.py` is the maintained entrypoint. The `Image Reader` and `PDF Reader` folders are preserved legacy examples.
- PDF files with embedded text do not require Tesseract. `--ocr` is only used when the PDF has no embedded text.
- Output defaults to a `.txt` file next to the input file and overwrites an existing file with that name.
- Files are processed locally. OCR and speech quality depend on the installed Tesseract data and local audio setup.
