# QR codes Readability

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **QR codes Readability**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Files in project directory:**

- `link_to_dataset.txt`

**Standardized data path:** `data/qr_codes_readability/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Evaluation metrics:**
- Accuracy

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Evaluate: Accuracy]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 11 |
| Code cells | 11 |
| Markdown cells | 0 |

## 6. Model Details

### Evaluation Metrics

- Accuracy

No model training in this project.

## 7. Project Structure

```
QR codes Readability/
├── qr-codes-readability.ipynb
├── link_to_dataset.txt
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `qr-codes-readability.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p156_*.py`:

```bash
python -m pytest tests/test_p156_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
