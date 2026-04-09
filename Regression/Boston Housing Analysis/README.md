# Boston Housing Analysis

## 1. Project Overview

This project implements a **Regression** pipeline for **Boston Housing Analysis**. The target variable is `Price`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `Price` |
| **Dataset Status** | DOWNLOADED |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `load_boston() (sklearn built-in)`

**Standardized data path:** `data/boston_housing_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load load_boston() (sklearn built-in)] --> B[Train/test split]
    B[Train/test split] --> C[LazyRegressor Benchmark]
    C[LazyRegressor Benchmark] --> D[PyCaret Regression]
    D[PyCaret Regression] --> E[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 33 |
| Code cells | 25 |
| Markdown cells | 8 |
| Original cells | 20 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

**⚠️ Deprecated APIs detected:**

- `load_boston()` is removed in scikit-learn ≥ 1.2 — this notebook will fail on modern sklearn

## 6. Model Details

### LazyRegressor (Standardized)

Compares 20+ regressors, ranked by RMSE/R².

### PyCaret Regression (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Boston Housing Analysis/
├── Ridge_And_Lasso_Regression.ipynb
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

- Open `Ridge_And_Lasso_Regression.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p003_*.py`:

```bash
python -m pytest tests/test_p003_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- `load_boston()` is removed in scikit-learn ≥ 1.2 — this notebook will fail on modern sklearn
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
