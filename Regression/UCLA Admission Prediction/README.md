# Predicting Admission into UCLA

## 1. Project Overview

This project implements a **Regression** pipeline for **Predicting Admission into UCLA**. The target variable is `Probability`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `Probability` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `admission_predict.csv`

**Files in project directory:**

- `admission_predict.csv`

**Standardized data path:** `data/predicting_admission_into_ucla/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load admission_predict.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Train/test split]
    C[Train/test split] --> D[LazyRegressor Benchmark]
    D[LazyRegressor Benchmark] --> E[PyCaret Regression]
    E[PyCaret Regression] --> F[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 39 |
| Code cells | 29 |
| Markdown cells | 10 |
| Original cells | 26 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

## 6. Model Details

### LazyRegressor (Standardized)

Compares 20+ regressors, ranked by RMSE/R².

### PyCaret Regression (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Predicting Admission into UCLA/
├── Predicting Admission into UCLA.ipynb
├── admission_predict.csv
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

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Predicting Admission into UCLA.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p068_*.py`:

```bash
python -m pytest tests/test_p068_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
