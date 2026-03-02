# Titanic Dataset - Exploratory Analysis

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Titanic Dataset - Exploratory Analysis**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `titan_train.csv`

**Standardized data path:** `data/titanic_dataset_-_exploratory_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop missing values (dropna)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load titan_train.csv] --> B[Drop missing values]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 67 |
| Code cells | 33 |
| Markdown cells | 34 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Titanic Dataset - Exploratory Analysis/
├── Titanic Dataset - Exploratory Analysis.ipynb
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

- Open `Titanic Dataset - Exploratory Analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p076_*.py`:

```bash
python -m pytest tests/test_p076_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
