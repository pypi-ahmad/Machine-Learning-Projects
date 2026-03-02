# TPOT Mercedes

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **TPOT Mercedes**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `mer_train.csv`

**Files in project directory:**

- `mercedesbenz.csv`

**Standardized data path:** `data/tpot_mercedes/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Train/test split

**Evaluation metrics:**
- Mean Squared Error
- Model Score

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load mer_train.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Train/test split]
    C[Train/test split] --> D[Evaluate: Mean Squared Error, Model Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 69 |
| Code cells | 41 |
| Markdown cells | 28 |

## 6. Model Details

### Evaluation Metrics

- Mean Squared Error
- Model Score

No model training in this project.

## 7. Project Structure

```
TPOT Mercedes/
├── TPOT Mercedes.ipynb
├── mercedesbenz.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `TPOT Mercedes.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p127_*.py`:

```bash
python -m pytest tests/test_p127_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
