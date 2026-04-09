# 911 Calls - Exploratory Analysis

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **911 Calls - Exploratory Analysis**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `911.csv`

**Standardized data path:** `data/911_calls_-_exploratory_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Date parsing
- Label mapping (function)
- Index setting

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load 911.csv] --> B[Date parsing]
    B[Date parsing] --> C[Label mapping]
    C[Label mapping] --> D[Index setting]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 59 |
| Code cells | 34 |
| Markdown cells | 25 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
911 Calls - Exploratory Analysis/
├── 911 Calls - Exploratory Analysis.ipynb
├── data
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `911 Calls - Exploratory Analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p071_*.py`:

```bash
python -m pytest tests/test_p071_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
