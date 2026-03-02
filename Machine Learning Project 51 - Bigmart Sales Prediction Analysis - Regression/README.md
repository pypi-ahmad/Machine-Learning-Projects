# Bigmart Sales Prediction Analysis - Regression

## 1. Project Overview

This project implements a **Regression** pipeline for **Bigmart Sales Prediction Analysis - Regression**. The target variable is `Item_Outlet_Sales`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `Item_Outlet_Sales` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Train.csv`

**Files in project directory:**

- `Train.csv`

**Standardized data path:** `data/bigmart_sales_prediction_analysis_-_regression/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Value replacement (dict-based)
- Label mapping (manual dict)
- Log transformation
- Label encoding (LabelEncoder)
- One-hot encoding (pd.get_dummies)
- Drop columns/rows

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Train.csv] --> B[Value replacement]
    B[Value replacement] --> C[Label mapping]
    C[Label mapping] --> D[Log transformation]
    D[Log transformation] --> E[Label encoding]
    E[Label encoding] --> F[One-hot encoding]
    F[One-hot encoding] --> G[Drop columns/rows]
    G[Drop columns/rows] --> H[LazyRegressor Benchmark]
    H[LazyRegressor Benchmark] --> I[PyCaret Regression]
    I[PyCaret Regression] --> J[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 68 |
| Code cells | 50 |
| Markdown cells | 18 |
| Original cells | 55 |
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
Bigmart Sales Prediction Analysis - Regression/
├── Bigmart Sales Prediction Analysis - Regression.ipynb
├── Train.csv
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

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Bigmart Sales Prediction Analysis - Regression.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p051_*.py`:

```bash
python -m pytest tests/test_p051_*.py -v
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
