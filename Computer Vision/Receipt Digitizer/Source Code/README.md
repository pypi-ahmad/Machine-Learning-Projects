# Receipt Digitizer

> **Task:** OCR + Field Extraction &nbsp;|&nbsp; **Key:** `receipt_digitizer` &nbsp;|&nbsp; **Framework:** EasyOCR

---

## Overview

Digitises receipts into structured, expense-ready data using EasyOCR for text detection/recognition, image preprocessing for noisy receipts, rule-based field parsing with confidence scores, validation rules, and JSON/CSV export. Supports batch inference over folders.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | OCR + Document AI |
| **Modern Stack** | EasyOCR (detection + recognition) + preprocessing + regex field parsing |
| **Dataset** | Receipt images ([jinhybr/OCR-receipt](https://huggingface.co/datasets/jinhybr/OCR-receipt) on Hugging Face) |
| **Key Metrics** | Field extraction accuracy, OCR recall |

## Dataset

- **Source:** Hugging Face — `jinhybr/OCR-receipt`
- **License:** See dataset page for license terms
- **Download:** Automatic on first `python train.py` run via `DatasetResolver`
- **Force re-download:** `python train.py --force-download`

## Project Structure

```
Receipt Digitizer/
└── Source Code/
    ├── config.py            # ReceiptConfig dataclass + YAML/JSON loader
    ├── preprocess.py        # Image cleanup: denoise, deskew, sharpen, binarize
    ├── ocr_engine.py        # EasyOCR wrapper → OCRBlock dataclass
    ├── parser.py            # Regex + heuristic field extraction → ParseResult
    ├── validator.py         # Validation rules + missing-field warnings
    ├── visualize.py         # Image overlay renderer
    ├── export.py            # JSON + CSV export with confidence scores
    ├── infer.py             # CLI pipeline (single image / batch directory)
    ├── modern.py            # CVProject subclass — @register("receipt_digitizer")
    ├── train.py             # Dataset download + OCR evaluation
    ├── data_bootstrap.py    # Dataset bootstrap via scripts/download_data.py
    ├── receipt_config.yaml  # Sample inference configuration
    ├── requirements.txt     # Project-level dependencies
    └── README.md            # This file
```

## Quick Start

### Inference (CLI)

```bash
cd "Receipt Digitizer/Source Code"

# Single image
python infer.py --source receipt.jpg

# Batch over directory
python infer.py --source receipts/ --export-json results.json --export-csv results.csv

# With preprocessing disabled
python infer.py --source receipt.jpg --no-preprocess

# Save annotated images, no GUI
python infer.py --source receipts/ --save-annotated --no-display --output-dir output

# GPU + custom language
python infer.py --source receipt.jpg --gpu --lang en
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("receipt_digitizer", "path/to/receipt.jpg")

print(result["result"].fields)       # extracted fields with confidence
print(result["report"].summary())    # validation report
```

### Training / Evaluation

```bash
cd "Receipt Digitizer/Source Code"
python train.py                    # download dataset + evaluate OCR
python train.py --force-download   # re-download dataset
python train.py --max-samples 50   # evaluate on more samples
```

## Extracted Fields

| Field | Description |
|-------|-------------|
| `merchant_name` | Store / restaurant name (first substantive line) |
| `date` | Transaction date |
| `time` | Transaction time |
| `subtotal` | Pre-tax subtotal |
| `tax` | Tax / VAT / GST amount |
| `tip` | Tip / gratuity (if present) |
| `total` | Grand total / amount due |
| `currency` | Detected currency symbol (USD, EUR, GBP, JPY) |
| `payment_method` | Payment hint (Visa, cash, etc.) |
| Line items | Description, quantity, unit price, amount |

## Preprocessing

Noisy receipt images are cleaned before OCR via configurable steps:

1. **Resize** — cap longest side (disabled by default)
2. **Deskew** — correct rotational skew via Hough lines
3. **Denoise** — `cv2.fastNlMeansDenoisingColored`
4. **Sharpen** — unsharp mask kernel
5. **Binarize** — adaptive threshold (disabled by default)

Each step is togglable in `receipt_config.yaml`.

## Dependencies

```
pip install easyocr opencv-python numpy pyyaml
```
