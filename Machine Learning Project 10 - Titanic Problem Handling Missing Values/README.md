# Titanic Problem Handling Missing Values

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Titanic Problem Handling Missing Values**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `titanic.csv`
- `titanic.csv`
- `titanic.csv`
- `mercedes.csv`
- `titanic.csv`
- `titanic.csv`
- `titanic.csv`
- `titanic.csv`
- `titanic.csv`

**Files in project directory:**

- `titanic.csv`

**Standardized data path:** `data/titanic_problem_handling_missing_values/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Handle missing values (fillna)
- Drop columns/rows
- One-hot encoding (pd.get_dummies)
- Drop missing values (dropna)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load titanic.csv] --> B[Handle missing values]
    B[Handle missing values] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[One-hot encoding]
    D[One-hot encoding] --> E[Drop missing values]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 130 |
| Code cells | 108 |
| Markdown cells | 22 |

**⚠️ Deprecated APIs detected:**

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Titanic Problem Handling Missing Values/
├── Missing _Value3.ipynb
├── Missing-Value1.ipynb
├── Missing_Value 2.ipynb
├── titanic.csv
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

- Open `Missing _Value3.ipynb` and run all cells
- Open `Missing-Value1.ipynb` and run all cells
- Open `Missing_Value 2.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p010_*.py`:

```bash
python -m pytest tests/test_p010_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`
- No model training — this is an analysis/tutorial notebook only
- Hardcoded file paths detected — may need adjustment
