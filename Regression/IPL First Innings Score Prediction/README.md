# First Innings Score Prediction - IPL

## 1. Project Overview

This project implements a **Regression** pipeline for **First Innings Score Prediction - IPL**. The target variable is `total`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `total` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `ipl.csv`

**Standardized data path:** `data/first_innings_score_prediction_-_ipl/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- One-hot encoding (pd.get_dummies)

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load ipl.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[One-hot encoding]
    C[One-hot encoding] --> D[LazyRegressor Benchmark]
    D[LazyRegressor Benchmark] --> E[PyCaret Regression]
    E[PyCaret Regression] --> F[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 40 |
| Code cells | 29 |
| Markdown cells | 11 |
| Original cells | 27 |
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
First Innings Score Prediction - IPL/
├── First Innings Score Prediction - IPL.ipynb
├── dataset
├── readme-resources
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

- Open `First Innings Score Prediction - IPL.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p066_*.py`:

```bash
python -m pytest tests/test_p066_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
