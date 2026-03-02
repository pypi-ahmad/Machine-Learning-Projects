# Black Friday Sales Prediction Analysis - Regression

## 1. Project Overview

This project implements a **Regression** pipeline for **Black Friday Sales Prediction Analysis - Regression**. The target variable is `Purchase`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `Purchase` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `train.csv`

**Files in project directory:**

- `train.csv`

**Standardized data path:** `data/black_friday_sales_prediction_analysis_-_regression/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Handle missing values (fillna)
- Data type conversion
- Label encoding (LabelEncoder)
- Drop columns/rows
- Train/test split

**Evaluation metrics:**
- Mean Squared Error
- Cross-Validation Score

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load train.csv] --> B[Handle missing values]
    B[Handle missing values] --> C[Data type conversion]
    C[Data type conversion] --> D[Label encoding]
    D[Label encoding] --> E[Drop columns/rows]
    E[Drop columns/rows] --> F[Train/test split]
    F[Train/test split] --> G[LazyRegressor Benchmark]
    G[LazyRegressor Benchmark] --> H[PyCaret Regression]
    H[PyCaret Regression] --> I[Evaluate: Mean Squared Error, Cross-Validation Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 48 |
| Code cells | 34 |
| Markdown cells | 14 |
| Original cells | 35 |
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

### Evaluation Metrics

- Mean Squared Error
- Cross-Validation Score

## 7. Project Structure

```
Black Friday Sales Prediction Analysis - Regression/
├── Black Friday Sales Prediction Analysis - Regression.ipynb
├── train.csv
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

- Open `Black Friday Sales Prediction Analysis - Regression.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p061_*.py`:

```bash
python -m pytest tests/test_p061_*.py -v
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
