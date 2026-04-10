# ID Card KYC Parser

> **Task:** Card Detection + OCR + KYC Field Extraction &nbsp;|&nbsp; **Key:** `id_card_kyc_parser` &nbsp;|&nbsp; **Framework:** PaddleOCR

---

## Overview

KYC-style parser that detects an ID card boundary in a photo, applies perspective correction to produce a front-facing crop, runs PaddleOCR, and extracts structured identity fields using template-based parsing. Supports multiple document types (generic ID, US driver licence, EU national ID, passport MRZ) with JSON/CSV export.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Card Detection + OCR + Document AI |
| **Card Detection** | Contour-based quadrilateral detection + four-point perspective transform |
| **OCR** | PaddleOCR (detection + recognition + angle classification) |
| **Field Parsing** | Template-based label-value matching with multi-language support |
| **Templates** | `generic`, `us_dl`, `eu_id`, `passport` (MRZ) |
| **Dataset** | FUNSD form/document images from Hugging Face |

## Dataset

- **Source:** Hugging Face — `nielsr/funsd` (FUNSD — Form Understanding in Noisy Scanned Documents)
- **License:** See dataset page for license terms
- **Download:** Automatic on first `python train.py` run via `DatasetResolver`
- **Force re-download:** `python train.py --force-download`

## Project Structure

```
ID Card KYC Parser/
└── Source Code/
    ├── config.py            # IDCardConfig dataclass + YAML/JSON loader
    ├── card_detector.py     # Card boundary detection + perspective rectification
    ├── ocr_engine.py        # PaddleOCR wrapper → OCRBlock dataclass
    ├── templates.py         # Isolated template definitions (generic, us_dl, eu_id, passport)
    ├── parser.py            # Template-agnostic parser orchestrator → ParseResult
    ├── validator.py         # Validation rules + missing-field warnings
    ├── visualize.py         # Colour-coded overlay with card boundary + KYC panel
    ├── export.py            # JSON + CSV export with confidence scores
    ├── infer.py             # CLI pipeline (single image / batch directory)
    ├── modern.py            # CVProject subclass — @register("id_card_kyc_parser")
    ├── train.py             # Dataset download + pipeline evaluation
    ├── data_bootstrap.py    # Dataset bootstrap via scripts/download_data.py
    ├── idcard_config.yaml   # Sample inference configuration
    ├── requirements.txt     # Project-level dependencies
    └── README.md            # This file
```

## Quick Start

### Inference (CLI)

```bash
cd "ID Card KYC Parser/Source Code"

# Single image — auto-detect card, rectify, extract fields
python infer.py --source id_card.jpg

# Batch with specific template
python infer.py --source cards/ --template passport --export-json kyc.json

# Skip card detection (pre-cropped images)
python infer.py --source cropped_card.jpg --no-detect

# Save both rectified crops and annotated overlays
python infer.py --source cards/ --save-annotated --save-rectified --no-display

# Export to CSV
python infer.py --source cards/ --export-csv contacts.csv --no-display
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("id_card_kyc_parser", "path/to/id_card.jpg")

print(result["detection"].found)      # card boundary detected?
print(result["result"].fields)        # extracted KYC fields
print(result["report"].summary())     # validation report
```

### Training / Evaluation

```bash
cd "ID Card KYC Parser/Source Code"
python train.py                          # download dataset + evaluate
python train.py --template passport      # evaluate with passport template
python train.py --force-download         # re-download dataset
python train.py --max-samples 50         # evaluate on more samples
```

## Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Input Image │───▶│  Card       │───▶│ PaddleOCR │───▶│ Template │───▶│ Validate │
│              │    │  Detector   │    │           │    │ Parser   │    │ + Export  │
│              │    │ (contour +  │    │ (text det │    │ (field   │    │          │
│              │    │  rectify)   │    │  + recog) │    │  extract)│    │          │
└─────────────┘    └─────────────┘    └───────────┘    └──────────┘    └──────────┘
```

## Templates

| Template | Document Type | Strategy |
|----------|---------------|----------|
| `generic` | Any ID card | Label-value pattern matching (multi-language) + fallback heuristics |
| `us_dl` | US Driver Licence | US-specific field labels (DL No., DOB, EXP, etc.) |
| `eu_id` | EU National ID | Multi-language EU labels (FR/ES/DE/EN) |
| `passport` | Passport | MRZ (Machine Readable Zone) TD-3 parsing |

## Extracted Fields

| Field | Description |
|-------|-------------|
| `full_name` | Person's full name |
| `date_of_birth` | Date of birth |
| `id_number` | Document / licence / passport number |
| `nationality` | Nationality / country |
| `gender` | Gender / sex |
| `expiry_date` | Document expiry date |
| `issue_date` | Document issue date |
| `address` | Address (if present) |
| `document_type` | Detected document type |

## Card Detection

The card detector uses Canny edge detection + contour approximation to find the largest quadrilateral in the image, then applies a four-point perspective transform to produce an axis-aligned, front-facing card crop at a standard resolution (856×540, matching ISO/IEC 7810 ID-1 proportions).

If no card boundary is found, the pipeline falls back to processing the full image with a warning.

## Dependencies

```
pip install paddleocr paddlepaddle opencv-python numpy pyyaml
```
