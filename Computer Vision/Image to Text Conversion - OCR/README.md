# Image to Text Conversion & Extraction - OCR

## 1. Project Overview

This project implements a **NLP / Text Analysis** pipeline for **Image to Text Conversion & Extraction - OCR**.

| Property | Value |
|----------|-------|
| **ML Task** | NLP / Text Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Standardized data path:** `data/image_to_text_conversion_extraction_-_ocr/`

## 3. Pipeline Overview

The original notebook primarily contains data loading and exploratory data analysis.

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Explore / Visualize]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 14 |
| Code cells | 10 |
| Markdown cells | 4 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Image to Text Conversion & Extraction - OCR/
├── Image to text (OCR).ipynb
├── test.JPG
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `Pillow`
- `matplotlib`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Image to text (OCR).ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p058_*.py`:

```bash
python -m pytest tests/test_p058_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
- Hardcoded file paths detected — may need adjustment
