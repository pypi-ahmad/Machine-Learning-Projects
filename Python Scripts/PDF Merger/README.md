# PDF Merger

Command-line utility for creating a new PDF by appending input PDFs in order, with optional page-position insertion.

## Setup

Requirements: Python 3.13+.

```powershell
cd "PDF Merger"
uv sync
```

## Usage

Append PDFs in order:

```powershell
uv run python merge_pdfs.py samplePdf1.pdf samplePdf2.pdf --output merged.pdf
```

Insert another PDF before page index `0` of the merged output:

```powershell
uv run python merge_pdfs.py samplePdf1.pdf --insert samplePdf2.pdf --insert-at 0 --output inserted.pdf
```

`--output` must name a new file. The command refuses to overwrite an existing PDF or use an input path as output.

## Behavior

1. Validates that every input file exists.
2. Appends positional input PDFs in the supplied order.
3. Optionally inserts one additional PDF at a zero-based page index.
4. Writes a new PDF using the maintained `pypdf` package.

Encrypted, malformed, or permission-restricted PDFs can fail to merge. This tool does not remove passwords, alter document content, or recover corrupted PDFs.

## Project files

```text
PDF Merger/
├── merge_pdfs.py
├── samplePdf1.pdf
├── samplePdf2.pdf
├── pyproject.toml
└── uv.lock
```
