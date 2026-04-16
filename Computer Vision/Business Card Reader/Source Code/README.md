# Business Card Reader

> **Task:** OCR + Contact Extraction &nbsp;|&nbsp; **Key:** `business_card_reader` &nbsp;|&nbsp; **Framework:** EasyOCR

---

## Overview

Extracts structured contact information from business card images using EasyOCR for text detection/recognition and rule-based field classification with confidence scores. Supports single-image and batch-folder modes with JSON/CSV export.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | OCR + Document AI |
| **Modern Stack** | EasyOCR (detection + recognition) + regex/heuristic field parsing |
| **Dataset** | Business card images (RVL-CDIP subset from Hugging Face) |
| **Key Metrics** | Field extraction accuracy, OCR recall |

## Dataset

- **Source:** Hugging Face — `aharley/rvl_cdip` (business card subset)
- **License:** See dataset page for license terms
- **Download:** Automatic on first `python train.py` run via `DatasetResolver`
- **Force re-download:** `python train.py --force-download`

## Project Structure

```
Business Card Reader/
└── Source Code/
    ├── config.py            # CardConfig dataclass + YAML/JSON loader
    ├── ocr_engine.py        # EasyOCR wrapper -> OCRBlock dataclass
    ├── parser.py            # Contact field extraction → ParseResult
    ├── validator.py         # Validation rules + missing-field warnings
    ├── visualize.py         # Colour-coded overlay renderer
    ├── export.py            # JSON + CSV export with confidence scores
    ├── infer.py             # CLI pipeline (single image / batch directory)
    ├── modern.py            # CVProject subclass — @register("business_card_reader")
    ├── train.py             # Dataset download + OCR evaluation
    ├── data_bootstrap.py    # Dataset bootstrap via scripts/download_data.py
    ├── card_config.yaml     # Sample inference configuration
    ├── requirements.txt     # Project-level dependencies
    └── README.md            # This file
```

## Quick Start

### Inference (CLI)

```bash
cd "Business Card Reader/Source Code"

# Single image
python infer.py --source card.jpg

# Batch over directory
python infer.py --source cards/ --export-json contacts.json --export-csv contacts.csv

# Save annotated images, no GUI
python infer.py --source cards/ --save-annotated --no-display --output-dir output

# GPU + custom language
python infer.py --source card.jpg --gpu --lang en
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("business_card_reader", "path/to/card.jpg")

print(result["result"].fields)       # extracted contact fields
print(result["report"].summary())    # validation report
```

### Training / Evaluation

```bash
cd "Business Card Reader/Source Code"
python train.py                    # download dataset + evaluate OCR
python train.py --force-download   # re-download dataset
python train.py --max-samples 50   # evaluate on more samples
```

## Extracted Fields

| Field | Strategy | Description |
|-------|----------|-------------|
| `name` | Residual | Person's name — most prominent unused alphabetical block |
| `title` | Keyword | Job title — matched via title keyword dictionary |
| `company` | Keyword + Heuristic | Company name — Corp/LLC/Inc suffixes or largest-font block |
| `phone` | Regex | Phone number — 7+ digit patterns with optional country code |
| `email` | Regex | Email address — standard email pattern matching |
| `website` | Regex | URL — www/http domain patterns (excluding email fragments) |
| `address` | Regex | Street address — matched via address indicator keywords and zip codes |

## Parsing Strategy

1. **Pattern-first:** Email, phone, website, and address are matched by strong regex patterns — these are the most reliable signals.
2. **Contextual:** Title is matched via a keyword dictionary; company via corporate suffixes or font-size heuristic.
3. **Residual:** After all pattern and contextual fields are consumed, the most prominent remaining alphabetical block is classified as the person's name.
4. **Colour-coded overlay:** Each field type gets a distinct colour in the visualization for easy verification.

## Dependencies

```
pip install easyocr opencv-python numpy pyyaml
```
