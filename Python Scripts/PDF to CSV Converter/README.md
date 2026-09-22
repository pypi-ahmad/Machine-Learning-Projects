# PDF to CSV Converter

Extract tables from one PDF into a CSV file with tabula-py.

## Setup

Install the Python dependencies:

```powershell
uv sync --no-config
```

tabula-py also requires a Java runtime. Install a JRE or JDK separately and make `java` available on `PATH`; `uv` does not install Java.

## Run

```powershell
uv run --no-config python main.py .\sample1.pdf
```

The default output is `sample1.csv` beside the input PDF. Supply an output path and a tabula-py page range when needed:

```powershell
uv run --no-config python main.py .\report.pdf --pages 2-5 --output .\report-tables.csv
```

Use `--help` for the available options.

## Behavior and limits

- Accepts one existing PDF per run and does not scan the working directory.
- Reads every page by default; `--pages` is passed to tabula-py.
- Overwrites the selected output path if tabula-py writes to it.
- Works best for PDFs with selectable, well-structured tables. It does not perform OCR on scanned image-only PDFs.
- Runs locally and makes no network requests.
