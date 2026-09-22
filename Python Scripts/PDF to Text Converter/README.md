# PDF to Text Converter

Extract embedded text from one PDF and save it as a UTF-8 text file.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python converter1.py .\samplePdf1.pdf
```

By default the output is placed beside the input PDF with a `.txt` extension. To choose a destination:

```powershell
uv run --no-config python converter1.py .\report.pdf --output .\report.txt
```

Use `--help` to view the available options.

## Behavior and limits

- Accepts one existing PDF per run; it does not prompt interactively or scan directories.
- Uses `pypdf` to read text already embedded in the PDF.
- Overwrites the selected output file if it already exists.
- Image-only or scanned PDFs need OCR and are not supported by this project.
- Runs locally and makes no network requests.

The bundled `samplePdf1.pdf`, `output.txt`, and `temp/` directory are preserved as legacy sample artifacts.
