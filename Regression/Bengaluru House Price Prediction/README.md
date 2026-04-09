# Predicting House Prices in Bengaluru

## 1. Project Overview

This project implements a **Regression** pipeline for **Predicting House Prices in Bengaluru**. The target variable is `price`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `price` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Bengaluru_House_Data.csv`

**Files in project directory:**

- `Bengaluru_House_Data.csv`

**Standardized data path:** `data/predicting_house_prices_in_bengaluru/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Handle missing values (fillna)
- Drop missing values (dropna)
- DataFrame concatenation
- One-hot encoding (pd.get_dummies)
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Bengaluru_House_Data.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Handle missing values]
    C[Handle missing values] --> D[Drop missing values]
    D[Drop missing values] --> E[DataFrame concatenation]
    E[DataFrame concatenation] --> F[One-hot encoding]
    F[One-hot encoding] --> G[Train/test split]
    G[Train/test split] --> H[LazyRegressor Benchmark]
    H[LazyRegressor Benchmark] --> I[PyCaret Regression]
    I[PyCaret Regression] --> J[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 65 |
| Code cells | 55 |
| Markdown cells | 10 |
| Original cells | 52 |
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
Predicting House Prices in Bengaluru/
├── Bengaluru house price prediction.ipynb
├── Bengaluru_House_Data.csv
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

- Open `Bengaluru house price prediction.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p069_*.py`:

```bash
python -m pytest tests/test_p069_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
