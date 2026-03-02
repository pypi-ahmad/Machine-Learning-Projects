# (Conceptual) nhanes_hypothesis_testing

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **(Conceptual) nhanes_hypothesis_testing**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `nhanes_2015_2016.csv`

**Files in project directory:**

- `NHANES.csv`

**Standardized data path:** `data/conceptual_nhanes_hypothesis_testing/`

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
| Total cells | 31 |
| Code cells | 16 |
| Markdown cells | 15 |

**⚠️ Deprecated APIs detected:**

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
(Conceptual) nhanes_hypothesis_testing/
├── nhanes_hypothesis_testing.ipynb
├── NHANES.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `scipy`
- `seaborn`
- `statsmodels`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `nhanes_hypothesis_testing.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p112_*.py`:

```bash
python -m pytest tests/test_p112_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`
- No model training — this is an analysis/tutorial notebook only
