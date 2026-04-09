# Boston Housing Prediction Analysis - Regression

## 1. Project Overview

This project implements a **Regression** pipeline for **Boston Housing Prediction Analysis - Regression**. The target variable is `medv`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `medv` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Boston Dataset.csv`

**Files in project directory:**

- `Boston Dataset.csv`

**Standardized data path:** `data/boston_housing_prediction_analysis_-_regression/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Feature scaling (StandardScaler)
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
    A[Load Boston Dataset.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Feature scaling]
    C[Feature scaling] --> D[Train/test split]
    D[Train/test split] --> E[LazyRegressor Benchmark]
    E[LazyRegressor Benchmark] --> F[PyCaret Regression]
    F[PyCaret Regression] --> G[Evaluate: Mean Squared Error, Cross-Validation Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 42 |
| Code cells | 24 |
| Markdown cells | 18 |
| Original cells | 29 |
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
Boston Housing Prediction Analysis - Regression/
├── Boston Housing Prediction - Regression.ipynb
├── Boston Dataset.csv
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

- Open `Boston Housing Prediction - Regression.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p060_*.py`:

```bash
python -m pytest tests/test_p060_*.py -v
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
