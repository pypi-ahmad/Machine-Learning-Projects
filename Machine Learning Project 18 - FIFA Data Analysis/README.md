# FIFA Data Analysis

## 1. Project Overview

This project implements a **Regression** pipeline for **FIFA Data Analysis**. The target variable is `Potential`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `Potential` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `data.csv`
- `data.csv`

**Files in project directory:**

- `data.csv`

**Standardized data path:** `data/fifa_data_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Drop missing values (dropna)
- Train/test split
- One-hot encoding (pd.get_dummies)

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load data.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Drop missing values]
    C[Drop missing values] --> D[Train/test split]
    D[Train/test split] --> E[One-hot encoding]
    E[One-hot encoding] --> F[LazyRegressor Benchmark]
    F[LazyRegressor Benchmark] --> G[PyCaret Regression]
    G[PyCaret Regression] --> H[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 62 |
| Code cells | 41 |
| Markdown cells | 21 |
| Original cells | 49 |
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
FIFA Data Analysis/
├── fifa-in-depth-analysis-with-linear-regression.ipynb
├── data.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `numpy`
- `pandas`
- `plotly`
- `pycaret`
- `scikit-learn`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `fifa-in-depth-analysis-with-linear-regression.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p018_*.py`:

```bash
python -m pytest tests/test_p018_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
