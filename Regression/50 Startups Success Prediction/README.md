# 50_startups Success Rate Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **50_startups Success Rate Prediction**. The target variable is `State`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `State` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Files in project directory:**

- `50_Startups.csv`

**Standardized data path:** `data/50_startups_success_rate_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- One-hot encoding (pd.get_dummies)
- Drop columns/rows
- DataFrame concatenation
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[One-hot encoding]
    B[One-hot encoding] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[DataFrame concatenation]
    D[DataFrame concatenation] --> E[Train/test split]
    E[Train/test split] --> F[LazyRegressor Benchmark]
    F[LazyRegressor Benchmark] --> G[PyCaret Regression]
    G[PyCaret Regression] --> H[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 26 |
| Code cells | 20 |
| Markdown cells | 6 |
| Original cells | 13 |
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
50_startups Success Rate Prediction/
├── Multiple_Linear_Regression.ipynb
├── 50_Startups.csv
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

- Open `Multiple_Linear_Regression.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p050_*.py`:

```bash
python -m pytest tests/test_p050_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Hardcoded file paths detected — may need adjustment

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
