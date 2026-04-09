# Bike Sharing Demand Analysis - Regression

## 1. Project Overview

This project implements a **Regression** pipeline for **Bike Sharing Demand Analysis - Regression**. The target variable is `count`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `count` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `hour.csv`

**Files in project directory:**

- `hour.csv`

**Standardized data path:** `data/bike_sharing_demand_analysis_-_regression/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Data type conversion
- Log transformation
- One-hot encoding (pd.get_dummies)
- DataFrame concatenation
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load hour.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Data type conversion]
    C[Data type conversion] --> D[Log transformation]
    D[Log transformation] --> E[One-hot encoding]
    E[One-hot encoding] --> F[DataFrame concatenation]
    F[DataFrame concatenation] --> G[Train/test split]
    G[Train/test split] --> H[LazyRegressor Benchmark]
    H[LazyRegressor Benchmark] --> I[PyCaret Regression]
    I[PyCaret Regression] --> J[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 47 |
| Code cells | 32 |
| Markdown cells | 15 |
| Original cells | 34 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

**⚠️ Deprecated APIs detected:**

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`

## 6. Model Details

### LazyRegressor (Standardized)

Compares 20+ regressors, ranked by RMSE/R².

### PyCaret Regression (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Bike Sharing Demand Analysis - Regression/
├── Bike Sharing Demand Analysis - Regression.ipynb
├── hour.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `numpy`
- `pandas`
- `pycaret`
- `scikit-learn`
- `seaborn`
- `statsmodels`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Bike Sharing Demand Analysis - Regression.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p062_*.py`:

```bash
python -m pytest tests/test_p062_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
