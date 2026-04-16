# Form OCR Checkbox Extractor

> **Task:** Checkbox Detection + OCR + Form Field Extraction &nbsp;|&nbsp; **Key:** `form_ocr_checkbox_extractor` &nbsp;|&nbsp; **Framework:** EasyOCR + OpenCV

---

## Overview

A form-understanding project that extracts typed/printed text and detects checkbox/radio-button states from scanned or photographed forms. Uses EasyOCR for text recognition and OpenCV morphological operations for checkbox/radio detection, then associates each control with its nearest text label and exports structured results to JSON/CSV.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Checkbox Detection + OCR + Form Understanding |
| **Checkbox Detection** | OpenCV adaptive threshold + morphology + contour analysis |
| **Radio Detection** | Circularity-based contour classification |
| **State Classification** | Interior pixel-fill ratio (Otsu threshold inside ROI) |
| **OCR** | EasyOCR (detection + recognition) |
| **Label Association** | Direction-weighted nearest-neighbour matching |
| **Dataset** | FUNSD — Form Understanding in Noisy Scanned Documents (Hugging Face) |

## Dataset

- **Source:** Hugging Face — `nielsr/funsd` (FUNSD)
- **License:** See dataset page for license terms
- **Download:** Automatic on first `python train.py` run via `DatasetResolver`
- **Force re-download:** `python train.py --force-download`

## Project Structure

```
Form OCR Checkbox Extractor/
└── Source Code/
    ├── config.py              # FormCheckboxConfig dataclass + YAML/JSON loader
    ├── checkbox_detector.py   # OpenCV checkbox/radio detection + state classification
    ├── ocr_engine.py          # EasyOCR wrapper -> OCRBlock dataclass
    ├── parser.py              # Form parser: OCR + checkbox + label association
    ├── validator.py           # Missing-field warnings + confidence checks
    ├── visualize.py           # Annotated overlay with checkbox outlines + panel
    ├── export.py              # JSON + CSV export with checkbox states
    ├── infer.py               # CLI pipeline (single image / batch directory)
    ├── modern.py              # CVProject subclass — @register("form_ocr_checkbox_extractor")
    ├── train.py               # Dataset download + pipeline evaluation
    ├── data_bootstrap.py      # Dataset bootstrap via scripts/download_data.py
    ├── form_config.yaml       # Sample inference configuration
    ├── requirements.txt       # Project-level dependencies
    └── README.md              # This file
```

## Quick Start

### Inference (CLI)

```bash
cd "Form OCR Checkbox Extractor/Source Code"

# Single form image
python infer.py --source form.jpg

# Batch processing with JSON export
python infer.py --source forms/ --export-json results.json --no-display

# Save annotated overlays
python infer.py --source forms/ --save-annotated --no-display

# Show OCR bounding boxes + export CSV
python infer.py --source form.jpg --show-ocr --export-csv forms.csv

# Custom fill threshold for checked/unchecked classification
python infer.py --source form.jpg --fill-threshold 0.40
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("form_ocr_checkbox_extractor", "path/to/form.jpg")

print(result["result"].num_checkboxes)    # total checkboxes found
print(result["result"].num_checked)       # how many are checked
print(result["result"].text_fields)       # extracted text fields
print(result["result"].checkbox_fields)   # checkbox labels + states
print(result["report"].summary())         # validation report
```

### Training / Evaluation

```bash
cd "Form OCR Checkbox Extractor/Source Code"
python train.py                              # download dataset + evaluate
python train.py --force-download             # re-download dataset
python train.py --max-samples 50             # evaluate on more samples
python train.py --fill-threshold 0.30        # tune checkbox sensitivity
```

## Pipeline

```
┌─────────────┐    ┌─────────────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
│  Input Image │───▶│  Checkbox       │───▶│ EasyOCR   │───▶│ Label      │───▶│ Validate │
│              │    │  Detector       │    │           │    │ Association│    │ + Export  │
│              │    │ (morphology +   │    │ (text det │    │ (nearest   │    │          │
│              │    │  contour)       │    │  + recog) │    │  neighbour)│    │          │
└─────────────┘    └─────────────────┘    └───────────┘    └────────────┘    └──────────┘
```

## Checkbox Detection

The detector uses an OpenCV-based approach:

1. **Adaptive thresholding** — converts to binary, handles uneven lighting
2. **Morphological closing** — cleans up noise and gaps in box edges
3. **Contour finding** — extracts all contours from the cleaned binary
4. **Size + aspect filtering** — keeps only small, roughly-square regions
5. **Type classification** — circularity ≥ 0.75 → radio button, else checkbox
6. **State classification** — Otsu threshold inside the ROI; if fill ratio ≥ 0.35 → checked
7. **De-duplication** — removes overlapping detections (IoU > 0.3)

### Tunables

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkbox_min_size` | 12 | Minimum side length (px) |
| `checkbox_max_size` | 60 | Maximum side length (px) |
| `fill_threshold` | 0.35 | Pixel-fill ratio to classify as checked |
| `radio_circularity` | 0.75 | Minimum circularity for radio buttons |
| `adaptive_block_size` | 25 | Block size for adaptive thresholding |

## Label Association

Each detected checkbox/radio is matched to its nearest OCR text block using a direction-weighted distance metric:

- **Preferred direction** (`right` by default) — labels to the right are preferred
- **Vertical penalty** — labels far above/below are penalised (3× vertical weight)
- **Maximum distance** — labels beyond 300px are not associated
- **Exclusive matching** — each text block is used at most once

## Validation

| Check | Severity | Description |
|-------|----------|-------------|
| Missing required fields | error | Required text fields not found |
| Low OCR confidence | warning | OCR confidence below 0.55 |
| Unlabelled checkboxes | warning | Checkboxes with no associated text |
| No checkboxes found | warning | No form controls detected |

## Export Format

### JSON

```json
{
  "total_forms": 1,
  "exported_at": "2026-04-09T12:00:00+00:00",
  "records": [
    {
      "source": "form.jpg",
      "text_fields": {
        "name": {"value": "John Doe", "confidence": 0.95}
      },
      "checkboxes": [
        {
          "label": "Agree to terms",
          "state": "checked",
          "type": "checkbox",
          "confidence": 0.92,
          "fill_ratio": 0.67
        }
      ],
      "num_checkboxes": 3,
      "num_checked": 1,
      "valid": true,
      "warnings": []
    }
  ]
}
```

## Dependencies

```
pip install easyocr opencv-python numpy pyyaml
```
