# (Conceptual) Regression Diagnostics

## 1. Project Overview

This project implements a **Regression** pipeline for **(Conceptual) Regression Diagnostics**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Files in project directory:**

- `Guerry.csv`

**Standardized data path:** `data/conceptual_regression_diagnostics/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Log transformation

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Log transformation]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 19 |
| Code cells | 11 |
| Markdown cells | 8 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
(Conceptual) Regression Diagnostics/
├── Regression Diagnostics.ipynb
├── Guerry.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `statsmodels`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Regression Diagnostics.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p118_*.py`:

```bash
python -m pytest tests/test_p118_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
