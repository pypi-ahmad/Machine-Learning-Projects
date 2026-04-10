# Invoice Field Extractor

> **Task:** OCR + Field Extraction &nbsp;|&nbsp; **Key:** `invoice_field_extractor` &nbsp;|&nbsp; **Framework:** PaddleOCR

---

## Overview

Extracts structured fields (invoice number, date, vendor, total, tax, line items) from invoice images and PDFs using PaddleOCR for text detection/recognition, rule-based field parsing with confidence scores, validation rules with missing-field warnings, and JSON/CSV export.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | OCR + Document AI |
| **Modern Stack** | PaddleOCR (detection + recognition) + regex field parsing |
| **Dataset** | Invoice document images (Hugging Face) |
| **Key Metrics** | Field extraction accuracy, OCR recall |

## Dataset

- **Source:** Hugging Face Datasets — invoice OCR / document understanding
- **License:** See dataset page for license terms
- **Download:** Automatic on first `python train.py` run via `DatasetResolver`
- **Force re-download:** `python train.py --force-download`

## Project Structure

```
Invoice Field Extractor/
└── Source Code/
    ├── config.py          # InvoiceConfig dataclass + YAML/JSON loader
    ├── data_bootstrap.py  # Dataset download via scripts/download_data.py
    ├── ocr_engine.py      # PaddleOCR wrapper → OCRBlock dataclass
    ├── parser.py          # Regex + heuristic field extraction → ParseResult
    ├── validator.py       # Validation rules + missing-field warnings
    ├── visualize.py       # Image overlay renderer
    ├── export.py          # JSON + CSV export with confidence scores
    ├── infer.py           # CLI pipeline (image / directory / PDF)
    ├── modern.py          # CVProject subclass — @register("invoice_field_extractor")
    ├── train.py           # Dataset download + OCR evaluation
    ├── requirements.txt   # Project-level dependencies
    └── README.md          # This file
```

## Quick Start

### Inference (CLI)

```bash
cd "Invoice Field Extractor/Source Code"

# Single image
python infer.py --source invoice.jpg

# Directory of images
python infer.py --source invoices/ --export-json results.json --export-csv results.csv

# PDF invoice
python infer.py --source invoice.pdf --no-display --save-annotated

# GPU + custom language
python infer.py --source invoice.jpg --gpu --lang en
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("invoice_field_extractor", "path/to/invoice.jpg")
print(result["result"].fields)       # extracted fields with confidence
print(result["report"].summary())    # validation report
```

### Training / Evaluation

```bash
cd "Invoice Field Extractor/Source Code"
python train.py                    # download dataset + evaluate OCR
python train.py --force-download   # re-download dataset
```

## Features

- **PaddleOCR** text detection with angle classification
- **Field extraction:** invoice number, date, vendor, total, subtotal, tax, currency, bill-to, line items
- **Field-level confidence scores** from OCR detection
- **Validation rules:** required field checks, total consistency, date format, low-confidence warnings
- **PDF support** via pdf2image or PyMuPDF
- **JSON + CSV export** with validation metadata
- **Annotated image overlay** with field highlights + summary panel
- **Batch processing** — directories and multi-page PDFs

## Dependencies

```
pip install paddleocr paddlepaddle opencv-python numpy pyyaml
```
