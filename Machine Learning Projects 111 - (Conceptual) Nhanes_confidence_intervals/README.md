# (Conceptual) Nhanes_confidence_intervals

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **(Conceptual) Nhanes_confidence_intervals**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `nhanes_2015_2016.csv`

**Files in project directory:**

- `NHANES.csv`

**Standardized data path:** `data/conceptual_nhanes_confidence_intervals/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Value replacement (dict-based)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load nhanes_2015_2016.csv] --> B[Value replacement]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 57 |
| Code cells | 27 |
| Markdown cells | 30 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
(Conceptual) Nhanes_confidence_intervals/
├── nhanes_confidence_intervals.ipynb
├── NHANES.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`
- `statsmodels`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `nhanes_confidence_intervals.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p111_*.py`:

```bash
python -m pytest tests/test_p111_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
