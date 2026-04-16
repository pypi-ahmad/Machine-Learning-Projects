# Prescription OCR Parser

> **Task:** Medical Prescription OCR + Structured Extraction &nbsp;|&nbsp; **Key:** `prescription_ocr_parser` &nbsp;|&nbsp; **Framework:** PaddleOCR-first OCR

---

> [!CAUTION]
> **MEDICAL SAFETY DISCLAIMER**
>
> This tool is for **informational and educational purposes only**.
> It does **NOT** provide medical advice, diagnosis, or treatment
> recommendations.  Extracted data may be incomplete, inaccurate,
> or misinterpreted by the OCR/parsing pipeline.
>
> **Do NOT** use this tool for clinical decision-making, dispensing
> medication, or any healthcare workflow without verification by a
> **licensed healthcare professional**.
>
> The authors accept no liability for any harm arising from the use
> or misuse of this software.

---

## Overview

A medical-document OCR parser that reads prescription images and extracts medicine-related text into structured output. The pipeline prefers PaddleOCR for text detection and recognition, then falls back to EasyOCR on local runtimes where PaddleOCR fails during inference. Pattern-based extraction turns OCR lines into medicine names, dosages, frequencies, routes, durations, and instruction-like notes. Outputs structured JSON/CSV with per-field confidence scores, low-confidence warnings, and annotated images.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Medical Document OCR + Structured Extraction |
| **OCR Engine** | PaddleOCR first, with automatic EasyOCR fallback on unsupported local runtimes |
| **Field Extraction** | Pattern-based: regex for dosages, keyword matching for frequency/route/duration |
| **Medicine Grouping** | Sequential line classification → medicine entry grouping |
| **Validation** | Missing dosage/frequency warnings, low-confidence alerts |
| **Dataset** | FUNSD document images from Hugging Face when available, otherwise synthetic prescription sheets generated locally |

## Dataset

- **Primary source:** Hugging Face — `nielsr/funsd` (FUNSD — Form Understanding in Noisy Scanned Documents)
- **License:** See the dataset page for the current upstream license terms.
- **Fallback:** If the public source is unavailable or does not yield usable images, the bootstrap code generates synthetic prescription samples locally for v1 validation.
- **Download:** Automatic on first `python train.py` or `python infer.py --force-download` run via `data_bootstrap.py`
- **Force re-download:** `python train.py --force-download`
- **Provenance file:** bootstrap writes `data/prescription_ocr_parser/dataset_info.json` with source and fallback metadata.

## Project Structure

```
Prescription OCR Parser/
└── Source Code/
    ├── config.py              # PrescriptionConfig dataclass + YAML/JSON loader
    ├── ocr_engine.py          # PaddleOCR-first wrapper → OCRBlock dataclass
    ├── field_extractor.py     # Pattern-based medicine + header field extraction
    ├── parser.py              # Pipeline orchestrator (OCR → extract → structure)
    ├── validator.py           # Missing-field + low-confidence validation
    ├── visualize.py           # Annotated overlay with medicine highlights + panel
    ├── export.py              # JSON + CSV export with medical disclaimer
    ├── infer.py               # CLI pipeline (single image / batch directory)
    ├── modern.py              # CVProject subclass — @register("prescription_ocr_parser")
    ├── train.py               # Dataset download + pipeline evaluation
    ├── data_bootstrap.py      # Dataset bootstrap with synthetic fallback
    ├── rx_config.yaml         # Sample inference configuration
    ├── requirements.txt       # Project-level dependencies
    └── README.md              # This file
```

## Quick Start

### Inference (CLI)

```bash
cd "Prescription OCR Parser/Source Code"

# Single prescription image
python infer.py --source prescription.jpg

# Batch with JSON export
python infer.py --source prescriptions/ --export-json results.json --no-display

# Export CSV (one row per medicine)
python infer.py --source prescriptions/ --export-csv medicines.csv --no-display

# Save annotated overlays
python infer.py --source prescriptions/ --save-annotated --no-display

# Show detailed OCR boxes
python infer.py --source rx.jpg --show-boxes
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("prescription_ocr_parser", "path/to/prescription.jpg")

print(result["result"].medicines)         # list of MedicineEntry
print(result["result"].header_fields)     # prescriber, patient, date
print(result["result"].num_medicines)     # count
print(result["report"].summary())         # validation report
```

### Training / Evaluation

```bash
cd "Prescription OCR Parser/Source Code"
python train.py                              # download dataset + evaluate
python train.py --force-download             # re-download dataset
python train.py --max-samples 50             # evaluate on more samples
```

## Pipeline

```
┌─────────────┐    ┌───────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐
│  Prescription│───▶│ PaddleOCR │───▶│  Field        │───▶│ Medicine │───▶│ Validate │
│  Image       │    │ / EasyOCR │    │  Extractor    │    │ Grouping │    │ + Export  │
│              │    │  fallback │    │ (dosage/freq/ │    │          │    │          │
│              │    │           │    │  route/instr) │    │          │    │          │
└─────────────┘    └───────────┘    └───────────────┘    └──────────┘    └──────────┘
```

## Extracted Fields

### Per Medicine

| Field | Detection Method | Example |
|-------|-----------------|---------|
| `medicine_name` | Alpha-dominant OCR line after header lines | Amoxicillin |
| `dosage` | Regex: `\d+ mg/mcg/ml/g/tablets/capsules` | 500mg |
| `frequency` | Keywords: daily, bid, tid, twice, every 8 hours | Twice daily |
| `duration` | Keywords + number: for X days/weeks | For 7 days |
| `route` | Keywords: oral, topical, iv, im, po | Oral |
| `instructions` | Keywords: take, apply, with food, as directed | Take with food |

### Header Fields

| Field | Detection Method |
|-------|-----------------|
| `prescriber` | Keywords: Dr, Doctor, MD, MBBS |
| `patient_name` | Keywords: Patient, Name, Mr/Mrs/Ms |
| `date` | Keywords + regex: `DD/MM/YYYY` patterns |

## Line Classification

Each OCR text line is classified into one of:

| Role | Criteria |
|------|----------|
| `medicine` | Alpha ratio ≥ 50%, length ≥ 3, not a keyword line |
| `dosage` | Matches dosage regex (number + unit) |
| `frequency` | Contains frequency keyword (daily, bid, etc.) |
| `duration` | Contains duration keyword + number |
| `route` | Contains route keyword (oral, topical, etc.) |
| `instruction` | Contains instruction keyword (take, apply, etc.) |
| `header` | Contains header keywords (Dr, Patient, Date) |

Lines are grouped sequentially: each `medicine` line starts a new entry, and subsequent detail lines attach to it.

## Validation

| Check | Severity | Description |
|-------|----------|-------------|
| No medicines found | error | No medicine entries detected |
| Missing dosage | warning | Medicine lacks dosage info |
| Missing frequency | warning | Medicine lacks timing info |
| Low OCR confidence | warning | Confidence below 0.50 |
| Missing header field | warning | Required header field not found |

## Export Format

### JSON

```json
{
  "disclaimer": "This output is for informational purposes only...",
  "total_prescriptions": 1,
  "exported_at": "2026-04-09T12:00:00+00:00",
  "records": [
    {
      "source": "rx.jpg",
      "header_fields": {
        "prescriber": {"value": "Dr. Smith", "confidence": 0.92}
      },
      "medicines": [
        {
          "medicine_name": "Amoxicillin",
          "dosage": "500mg",
          "frequency": "Twice daily",
          "duration": "For 7 days",
          "route": "Oral",
          "instructions": "Take with food",
          "confidence": 0.88
        }
      ]
    }
  ]
}
```

### CSV

One row per medicine entry with columns: `source`, `medicine_name`, `dosage`, `frequency`, `duration`, `route`, `instructions`, `confidence`, `prescriber`, `patient_name`, `date`, `valid`.

## Limitations

- **Pattern-based extraction** — relies on common English medical abbreviations and keywords. May miss non-standard or handwritten prescriptions with unusual formatting.
- **No drug database lookup** — does not validate medicine names against a pharmacological database.
- **No handwriting model** — this v1 targets printed or clearly scanned prescription-style documents. For handwritten prescriptions, pair with a specialised handwriting recogniser.
- **Proxy dataset** — v1 uses FUNSD document images. Accuracy will improve with a dedicated prescription dataset.
- **Runtime note** — this Windows environment can trigger a PaddleOCR oneDNN runtime error during inference; the project automatically falls back to EasyOCR so the pipeline remains usable.

## Dependencies

```bash
pip install paddleocr paddlepaddle easyocr opencv-python numpy pyyaml
```
