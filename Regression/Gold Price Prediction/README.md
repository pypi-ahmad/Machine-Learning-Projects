# Gold Price Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **Gold Price Prediction**. The target variable is `Return`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `Return` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `gold_price.csv`

**Files in project directory:**

- `gold_price.csv`

**Standardized data path:** `data/gold_price_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Percentage change calculation
- Feature lagging (shift)
- Drop missing values (dropna)

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load gold_price.csv] --> B[Percentage change calculation]
    B[Percentage change calculation] --> C[Feature lagging]
    C[Feature lagging] --> D[Drop missing values]
    D[Drop missing values] --> E[LazyRegressor Benchmark]
    E[LazyRegressor Benchmark] --> F[PyCaret Regression]
    F[PyCaret Regression] --> G[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 15 |
| Code cells | 9 |
| Markdown cells | 6 |
| Original cells | 2 |
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
Gold Price Prediction/
├── Untitled.ipynb
├── gold_price.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `pycaret`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Untitled.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p013_*.py`:

```bash
python -m pytest tests/test_p013_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Notebook uses default name (`Untitled.ipynb`)

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
